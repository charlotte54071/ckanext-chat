import asyncio
import multiprocessing as mp
import os
import sys
from distutils.util import strtobool
from typing import Any

import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, current_app, jsonify, request
from flask.views import MethodView
from loguru import logger
from pydantic_ai.messages import TextPart

# from ckanext.chat.bot.agent import (Deps, async_agent_response,
#                                     exception_to_model_response,
#                                     user_input_to_model_request)
from ckanext.chat.bot.agent import (exception_to_model_response,
                                    user_input_to_model_request)
from ckanext.chat.helpers import service_available
import json
from flask import Response


#mp.set_start_method("spawn", force=True)
logger.remove()
if bool(strtobool(os.environ.get("DEBUG", "false"))):
    log_level = "DEBUG"
else:
    log_level = "ERROR"
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | [{name}] {message}",
    level=log_level,
    enqueue=True,
)

blueprint = Blueprint("chat", __name__)

global_ckan_app = None


@blueprint.before_request
def capture_global_app():
    # This hook is executed in an active application context.
    global global_ckan_app
    if global_ckan_app is None:
        # Capture the global CKAN app from the current request's context
        global_ckan_app = current_app._get_current_object()


class ChatView(MethodView):
    def post(self):
        return core_helpers.redirect_to(
            "chat.chat",
        )

    def get(self):
        if current_user.is_anonymous:
            core_helpers.flash_error(_("Not authorized to see this page"))

            # flask types do not mention that it's possible to return a response
            # from the `before_request` callback
            return core_helpers.redirect_to("user.login")
        # logger.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )

def to_jsonable(obj):
    # Ckan result
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)

def serialize_part(part):
    # pydantic-ai common part types

    try:
        d = {}
        if hasattr(part, "content"):
            d["type"] = type(part).__name__
            d["content"] = part.content
            return d
        if hasattr(part, "url"):
            return {"type": type(part).__name__, "url": str(part.url)}
        return to_jsonable(part)
    except Exception:
        return to_jsonable(part)

def serialize_message(msg):
    try:
        data = {
            "type": type(msg).__name__,
            "model_name": getattr(msg, "model_name", None),
            "kind": getattr(msg, "kind", None),
            "timestamp": getattr(msg, "timestamp", None).isoformat()
                if getattr(msg, "timestamp", None) else None,
            "parts": [serialize_part(p) for p in getattr(msg, "parts", [])],
        }
        usage = getattr(msg, "usage", None)
        if usage is not None:
            data["usage"] = to_jsonable(usage)
        return data
    except Exception:
        return to_jsonable(msg)

def serialize_messages(messages):
    # josn
    return [serialize_message(m) for m in messages]

def ask():
    logger.debug(request.form)
    user_input = request.form.get("text")
    history = request.form.get("history", "")

    # 兼容拼写：前端可能传 reserach / research 任意一个
    research_raw = request.form.get("reserach", request.form.get("research", False))
    # 归一化为布尔
    research = str(research_raw).lower() in {"1", "true", "yes", "on"}

    max_retries = 3
    attempt = 0
    tkuser = toolkit.current_user
    debug = bool(strtobool(os.environ.get("DEBUG", "false")))

    if tkuser.name is None:
        payload = {"success": False, "msg": "Must be logged in to view site"}
        return Response(json.dumps(payload, ensure_ascii=False), mimetype="application/json", status=401)

    while attempt < max_retries:
        try:
            response = asyncio.run(
                async_agent_response(user_input, history, user_id=tkuser.id, research=research),
                debug=debug,
            )

            messages = response.new_messages()

            # remove none
            for message in messages:
                parts_copy = list(getattr(message, "parts", []))
                for part in parts_copy:
                    if isinstance(part, TextPart) and getattr(part, "content", "") == "":
                        message.parts.remove(part)

            payload = {"response": serialize_messages(messages)}
            return Response(
                json.dumps(payload, default=to_jsonable, ensure_ascii=False),
                mimetype="application/json",
                status=200,
            )

        except Exception as e:
            user_prompt = user_input_to_model_request(user_input)
            error_response = exception_to_model_response(e)
            logger.error(error_response)

            payload = {"response": serialize_messages([user_prompt, error_response])}
            return Response(
                json.dumps(payload, default=to_jsonable, ensure_ascii=False),
                mimetype="application/json",
                status=200,
            )

def async_agent_response(prompt: str, history: str, user_id: str, research: bool = False) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_agent_worker(prompt, history, user_id, research))
    finally:
        loop.close()

async def _agent_worker(prompt: str, history: str, user_id: str, research: bool = False) -> Any:
    from loguru import logger
    from ckanext.chat.bot.agent import (
        Deps, agent, research_agent, convert_to_model_messages
    )
    from ckanext.chat.bot.utils import init_dynamic_models, dynamic_models_initialized

    logger = logger.bind(process="worker", user_id=user_id)
    logger.debug(f"Worker starting for {user_id}")

    if not dynamic_models_initialized:
        init_dynamic_models()

    deps = Deps(user_id=user_id)
    msg_history = convert_to_model_messages(history)

    if research:
        r = research_agent.run(
            user_prompt=prompt,
            message_history=msg_history,
            deps=deps,
        )
    else:
        r = agent.run(
            user_prompt=prompt,
            message_history=msg_history,
            deps=deps,
        )

    logger.debug(f"Worker done, result: {r}")
    await logger.complete()
    return r



blueprint.add_url_rule(
    "/chat",
    view_func=ChatView.as_view(str("chat")),
)

blueprint.add_url_rule(
    "/chat/ask",
    view_func=ask,
    methods=["POST"],
)


def get_blueprint():
    return blueprint
