from flask import Blueprint
from flask.views import MethodView
from ckanext.chat.helpers import service_available
from ckan.common import _, current_user
import ckan.lib.helpers as core_helpers
import ckan.lib.base as base

blueprint = Blueprint("chat", __name__)


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
            },
        )


blueprint.add_url_rule(
    "/chat",
    view_func=ChatView.as_view(str("chat")),
)


def get_blueprint():
    return blueprint
