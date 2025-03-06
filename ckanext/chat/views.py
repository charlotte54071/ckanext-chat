import asyncio
import json
from datetime import datetime

import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, Response, jsonify, request, current_app
from flask.views import MethodView
from pydantic_ai.exceptions import UnexpectedModelBehavior

from ckanext.chat.bot import agent, prompt
from ckanext.chat.bot.agent import agent_response, get_ckan_url_patterns
from ckanext.chat.bot.code_generator import CodeGenerator
from ckanext.chat.helpers import service_available

blueprint = Blueprint("chat", __name__)

log = __import__("logging").getLogger(__name__)
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
        #log.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )


def ask():
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    log.debug(history)

    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            response = agent_response(user_input, history)
            return jsonify({"response": response.all_messages()})
        except Exception as e:
            log.error(f"Attempt {attempt + 1}: {e}")
            attempt += 1
            

    # Inform the user if all retries fail
    return (
        jsonify(
            {
                "error": "Failed to get a valid response from the AI model after multiple attempts."
            }
        ),
        500,
    )


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
