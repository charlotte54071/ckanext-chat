from flask import Blueprint, request, jsonify, session
from flask.views import MethodView

from ckanext.chat.bot import pritgpt, prompt
from ckanext.chat.bot.code_generator import CodeGenerator
from ckanext.chat.bot.pritgpt import qagpt_response

from ckanext.chat.helpers import service_available
from ckan.common import _, current_user
import ckan.lib.helpers as core_helpers
import ckan.lib.base as base
import ckan.plugins.toolkit as toolkit

import datetime

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
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )


def ask():
    user_input = request.form["text"]
    response = chatbot_db_query_process(user_input)
    return jsonify({"response": response})


def chatbot_db_query_process(user_input):
    python_or_gen = prompt.python_or_general_response(user_input)
    add_to_session_history("user", user_input)
    response = qagpt_response(
        [{"role": "user", "content": python_or_gen}],
        model="llama3-8b-8192",
        type="ollama",
        temperature=0.90,
    )
    if "true" in response.choices[0].message.content.lower():
        print("In true case: " + response.choices[0].message.content.lower())
        code = CodeGenerator(prompt=user_input)
        gen_code = code.generate_code()
        final_response = code.debug_and_execute(gen_code)
        return final_response
    else:
        print(
            "In false case with db enabled: "
            + response.choices[0].message.content.lower()
        )
        print("session history:", session["history"][-10:])
        # normal action
        response = qagpt_response(
            session["history"][-10:],
            model="llama3-8b-8192",
            type="ollama",
            temperature=0.50,
        )
        add_to_session_history("assistant", response.choices[0].message.content)
        final_response = response.choices[0].message.content

        return final_response


def add_to_session_history(role, content):
    """Add a message to the session history."""
    if "history" not in session:
        session["history"] = [
            {
                "role": "system",
                "content": f'Today is {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}! Always think before you answer.',
            }
        ]
    session["history"].append({"role": role, "content": content})
    session.modified = True


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
