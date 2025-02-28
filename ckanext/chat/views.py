from flask import Blueprint, request, jsonify, session
from flask.views import MethodView

from ckanext.chat.bot import agent, prompt
from ckanext.chat.bot.code_generator import CodeGenerator
from ckanext.chat.bot.agent import agent_response

from ckanext.chat.helpers import service_available
from ckan.common import _, current_user
import ckan.lib.helpers as core_helpers
import ckan.lib.base as base
import ckan.plugins.toolkit as toolkit
import asyncio


blueprint = Blueprint("chat", __name__)

log = __import__("logging").getLogger(__name__)


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
    response = asyncio.run(agent_response(user_input, history))
    return jsonify({"response": response.all_messages()})


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
