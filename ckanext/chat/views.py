import asyncio
import multiprocessing as mp
from logging.handlers import QueueListener
import logging, os
from distutils.util import strtobool 
from typing import Any

import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, current_app, jsonify, request
from flask.views import MethodView

# from ckanext.chat.bot.agent import (Deps, async_agent_response,
#                                     exception_to_model_response,
#                                     user_input_to_model_request)
from ckanext.chat.bot.agent import (exception_to_model_response,
                                    user_input_to_model_request)

from ckanext.chat.helpers import service_available

blueprint = Blueprint("chat", __name__)

# configure logging for main thread nd workers
log = __import__("logging").getLogger(__name__)

# 1. Create a shared queue
log_queue = mp.get_context("spawn").Queue(-1)

# Main handler configuration (console or file)
handler = logging.StreamHandler()
if bool(strtobool(os.environ.get('DEBUG','false'))):
    log.debug('enabling DEBUG Logging on ckanext.chat and its workers')
    handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    "%Y-%m-%d %H:%M:%S,%f"
))

# Listener applies the formatting
queue_listener = QueueListener(
    log_queue,
    handler,
    respect_handler_level=True
)
queue_listener.start()

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
        # log.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )


from pydantic_ai.messages import TextPart

def ask():
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    max_retries = 3
    attempt = 0
    tkuser = toolkit.current_user
    # If they're not a logged in user, don't allow them to see content
    if tkuser.name is None:
        return {"success": False, "msg": "Must be logged in to view site"}
    while attempt < max_retries:
        try:
            response = asyncio.run(async_agent_response(user_input, history, user_id=tkuser.id), debug=True)
            # Now response is guaranteed to have new_messages() if no exception occurred.
            # Ensure new_messages() is awaited in the sync wrapper if it's async
            messages = response.new_messages()
            # remove empty text responses parts
            [
                [
                    message.parts.remove(part)
                    for part in message.parts
                    if isinstance(part, TextPart) and part.content == ""
                ]
                for message in messages
            ]
            return jsonify({"response": messages})

        except Exception as e:
            user_promt = user_input_to_model_request(user_input)
            error_response = exception_to_model_response(e)
            log.error(error_response)
            return jsonify({"response": [user_promt, error_response]})


async def async_agent_response(prompt: str, history: str, user_id: str) -> Any:
    return await _agent_worker(prompt, history, user_id)


async def _agent_worker(prompt: str, history: str, user_id: str) -> Any:
    from logging.handlers import QueueHandler
    # Attach queue handler to your module's logger
    mod_logger = logging.getLogger("ckanext.chat.bot.agent")
    mod_logger.handlers.clear()
    mod_logger.setLevel(logging.DEBUG)
    mod_logger.propagate = False
    mod_logger.addHandler(QueueHandler(log_queue))

    # Silence other libraries
    for noisy in ("httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    mod_logger.debug("Worker starting for user %s", user_id)

    from ckanext.chat.bot.agent import agent, init_dynamic_models, convert_to_model_messages, Deps, dynamic_models_initialized
    if not dynamic_models_initialized:
        init_dynamic_models()
    deps = deps = Deps(user_id=user_id)
    msg_history = convert_to_model_messages(history)
    # Run the async agent
    result = await agent.run(
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
    )
    return result
    

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
