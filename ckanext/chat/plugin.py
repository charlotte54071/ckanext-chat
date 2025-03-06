import ckan.plugins as plugins
import ckan.plugins.toolkit as toolkit
from ckan.config.declaration import Declaration, Key
from ckanext.chat import action, auth, helpers, views
from ckan.types import Schema

from flask import has_request_context,current_app
class ChatPlugin(plugins.SingletonPlugin):
    plugins.implements(plugins.IConfigurer)
    plugins.implements(plugins.IConfigDeclaration)
    plugins.implements(plugins.IBlueprint)
    plugins.implements(plugins.ITemplateHelpers)
    plugins.implements(plugins.IAuthFunctions)
    plugins.implements(plugins.IActions)
    #plugins.implements(plugins.IConfigurable)

    # IConfigurer

    def update_config(self, config_):
        toolkit.add_template_directory(config_, "templates")
        toolkit.add_public_directory(config_, "public")
        toolkit.add_resource("assets", "chat")

    # IConfigDeclaration

    def declare_config_options(self, declaration: Declaration, key: Key):

        declaration.annotate("chat")
        group = key.ckanext.chat
        declaration.declare_bool(group.ssl_verify, True)
        declaration.declare(group.completion_url, "https://your.chat.api")
        declaration.declare(group.deployment, "gpt-4-vision-preview")
        declaration.declare(group.api_token, "")
        log = __import__("logging").getLogger(__name__)

    
    # IBlueprint

    def get_blueprint(self):
        return views.get_blueprint()

    # ITemplateHelpers

    def get_helpers(self):
        return helpers.get_helpers()

    # IAuthFunctions

    def get_auth_functions(self):
        return auth.get_auth_functions()

    # IActions

    def get_actions(self):
        actions = action.get_actions()
        return actions
