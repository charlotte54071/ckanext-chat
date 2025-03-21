import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, current_app, jsonify, request
from flask.views import MethodView

from ckanext.chat.bot.agent import (
    Deps,
    agent_response,
    exception_to_model_response,
    user_input_to_model_request,
)
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


def ask():
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    max_retries = 3
    attempt = 0
    tkuser = toolkit.current_user
    # If they're not a logged in user, don't allow them to see content
    if tkuser.name is None:
        return {"success": False, "msg": "Must be logged in to view site"}
    deps = Deps(user_id=tkuser.id)
    # log.debug(user)
    while attempt < max_retries:
        try:
            response = agent_response(user_input, history, deps=deps)
            # Now response is guaranteed to have new_messages() if no exception occurred.
            return jsonify({"response": response.new_messages()})
        except Exception as e:
            user_promt = user_input_to_model_request(user_input)
            error_response = exception_to_model_response(e)
            log.error(error_response)
            return jsonify({"response": [user_promt, error_response]})


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
