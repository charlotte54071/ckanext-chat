import logging

import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, current_app, flash, jsonify, request
from flask.views import MethodView
from pydantic_ai.exceptions import (AgentRunError, FallbackExceptionGroup,
                                    ModelHTTPError, ModelRetry,
                                    UnexpectedModelBehavior,
                                    UsageLimitExceeded)

from ckanext.chat.bot.agent import agent_response, exception_to_model_response
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
        # log.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )


from ckanext.chat.bot.agent import CKANUser


def ask():
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    max_retries = 3
    attempt = 0
    tkuser = toolkit.current_user
    # If they're not a logged in user, don't allow them to see content
    if tkuser.name is None:
        return {'success': False,
                'msg': 'Must be logged in to view site'}
    user = CKANUser(id=tkuser.id, name=tkuser.name)
    log.debug(user)
    while attempt < max_retries:
        try:
            response = agent_response(user, user_input, history)
            # Now response is guaranteed to have new_messages() if no exception occurred.
            return jsonify({"response": response.new_messages()})
        except Exception as e:
            error_response = exception_to_model_response(e)
            log.error(error_response)
            return jsonify({"response": [error_response,]})

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
