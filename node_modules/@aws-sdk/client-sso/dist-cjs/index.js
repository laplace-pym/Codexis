'use strict';

var middlewareHostHeader = require('@aws-sdk/middleware-host-header');
var middlewareLogger = require('@aws-sdk/middleware-logger');
var middlewareRecursionDetection = require('@aws-sdk/middleware-recursion-detection');
var middlewareUserAgent = require('@aws-sdk/middleware-user-agent');
var configResolver = require('@smithy/config-resolver');
var core = require('@smithy/core');
var schema = require('@smithy/core/schema');
var middlewareContentLength = require('@smithy/middleware-content-length');
var middlewareEndpoint = require('@smithy/middleware-endpoint');
var middlewareRetry = require('@smithy/middleware-retry');
var smithyClient = require('@smithy/smithy-client');
var httpAuthSchemeProvider = require('./auth/httpAuthSchemeProvider');
var runtimeConfig = require('./runtimeConfig');
var regionConfigResolver = require('@aws-sdk/region-config-resolver');
var protocolHttp = require('@smithy/protocol-http');

const resolveClientEndpointParameters = (options) => {
    return Object.assign(options, {
        useDualstackEndpoint: options.useDualstackEndpoint ?? false,
        useFipsEndpoint: options.useFipsEndpoint ?? false,
        defaultSigningName: "awsssoportal",
    });
};
const commonParams = {
    UseFIPS: { type: "builtInParams", name: "useFipsEndpoint" },
    Endpoint: { type: "builtInParams", name: "endpoint" },
    Region: { type: "builtInParams", name: "region" },
    UseDualStack: { type: "builtInParams", name: "useDualstackEndpoint" },
};

const getHttpAuthExtensionConfiguration = (runtimeConfig) => {
    const _httpAuthSchemes = runtimeConfig.httpAuthSchemes;
    let _httpAuthSchemeProvider = runtimeConfig.httpAuthSchemeProvider;
    let _credentials = runtimeConfig.credentials;
    return {
        setHttpAuthScheme(httpAuthScheme) {
            const index = _httpAuthSchemes.findIndex((scheme) => scheme.schemeId === httpAuthScheme.schemeId);
            if (index === -1) {
                _httpAuthSchemes.push(httpAuthScheme);
            }
            else {
                _httpAuthSchemes.splice(index, 1, httpAuthScheme);
            }
        },
        httpAuthSchemes() {
            return _httpAuthSchemes;
        },
        setHttpAuthSchemeProvider(httpAuthSchemeProvider) {
            _httpAuthSchemeProvider = httpAuthSchemeProvider;
        },
        httpAuthSchemeProvider() {
            return _httpAuthSchemeProvider;
        },
        setCredentials(credentials) {
            _credentials = credentials;
        },
        credentials() {
            return _credentials;
        },
    };
};
const resolveHttpAuthRuntimeConfig = (config) => {
    return {
        httpAuthSchemes: config.httpAuthSchemes(),
        httpAuthSchemeProvider: config.httpAuthSchemeProvider(),
        credentials: config.credentials(),
    };
};

const resolveRuntimeExtensions = (runtimeConfig, extensions) => {
    const extensionConfiguration = Object.assign(regionConfigResolver.getAwsRegionExtensionConfiguration(runtimeConfig), smithyClient.getDefaultExtensionConfiguration(runtimeConfig), protocolHttp.getHttpHandlerExtensionConfiguration(runtimeConfig), getHttpAuthExtensionConfiguration(runtimeConfig));
    extensions.forEach((extension) => extension.configure(extensionConfiguration));
    return Object.assign(runtimeConfig, regionConfigResolver.resolveAwsRegionExtensionConfiguration(extensionConfiguration), smithyClient.resolveDefaultRuntimeConfig(extensionConfiguration), protocolHttp.resolveHttpHandlerRuntimeConfig(extensionConfiguration), resolveHttpAuthRuntimeConfig(extensionConfiguration));
};

class SSOClient extends smithyClient.Client {
    config;
    constructor(...[configuration]) {
        const _config_0 = runtimeConfig.getRuntimeConfig(configuration || {});
        super(_config_0);
        this.initConfig = _config_0;
        const _config_1 = resolveClientEndpointParameters(_config_0);
        const _config_2 = middlewareUserAgent.resolveUserAgentConfig(_config_1);
        const _config_3 = middlewareRetry.resolveRetryConfig(_config_2);
        const _config_4 = configResolver.resolveRegionConfig(_config_3);
        const _config_5 = middlewareHostHeader.resolveHostHeaderConfig(_config_4);
        const _config_6 = middlewareEndpoint.resolveEndpointConfig(_config_5);
        const _config_7 = httpAuthSchemeProvider.resolveHttpAuthSchemeConfig(_config_6);
        const _config_8 = resolveRuntimeExtensions(_config_7, configuration?.extensions || []);
        this.config = _config_8;
        this.middlewareStack.use(schema.getSchemaSerdePlugin(this.config));
        this.middlewareStack.use(middlewareUserAgent.getUserAgentPlugin(this.config));
        this.middlewareStack.use(middlewareRetry.getRetryPlugin(this.config));
        this.middlewareStack.use(middlewareContentLength.getContentLengthPlugin(this.config));
        this.middlewareStack.use(middlewareHostHeader.getHostHeaderPlugin(this.config));
        this.middlewareStack.use(middlewareLogger.getLoggerPlugin(this.config));
        this.middlewareStack.use(middlewareRecursionDetection.getRecursionDetectionPlugin(this.config));
        this.middlewareStack.use(core.getHttpAuthSchemeEndpointRuleSetPlugin(this.config, {
            httpAuthSchemeParametersProvider: httpAuthSchemeProvider.defaultSSOHttpAuthSchemeParametersProvider,
            identityProviderConfigProvider: async (config) => new core.DefaultIdentityProviderConfig({
                "aws.auth#sigv4": config.credentials,
            }),
        }));
        this.middlewareStack.use(core.getHttpSigningPlugin(this.config));
    }
    destroy() {
        super.destroy();
    }
}

class SSOServiceException extends smithyClient.ServiceException {
    constructor(options) {
        super(options);
        Object.setPrototypeOf(this, SSOServiceException.prototype);
    }
}

class InvalidRequestException extends SSOServiceException {
    name = "InvalidRequestException";
    $fault = "client";
    constructor(opts) {
        super({
            name: "InvalidRequestException",
            $fault: "client",
            ...opts,
        });
        Object.setPrototypeOf(this, InvalidRequestException.prototype);
    }
}
class ResourceNotFoundException extends SSOServiceException {
    name = "ResourceNotFoundException";
    $fault = "client";
    constructor(opts) {
        super({
            name: "ResourceNotFoundException",
            $fault: "client",
            ...opts,
        });
        Object.setPrototypeOf(this, ResourceNotFoundException.prototype);
    }
}
class TooManyRequestsException extends SSOServiceException {
    name = "TooManyRequestsException";
    $fault = "client";
    constructor(opts) {
        super({
            name: "TooManyRequestsException",
            $fault: "client",
            ...opts,
        });
        Object.setPrototypeOf(this, TooManyRequestsException.prototype);
    }
}
class UnauthorizedException extends SSOServiceException {
    name = "UnauthorizedException";
    $fault = "client";
    constructor(opts) {
        super({
            name: "UnauthorizedException",
            $fault: "client",
            ...opts,
        });
        Object.setPrototypeOf(this, UnauthorizedException.prototype);
    }
}

const _AI = "AccountInfo";
const _ALT = "AccountListType";
const _ATT = "AccessTokenType";
const _GRC = "GetRoleCredentials";
const _GRCR = "GetRoleCredentialsRequest";
const _GRCRe = "GetRoleCredentialsResponse";
const _IRE = "InvalidRequestException";
const _L = "Logout";
const _LA = "ListAccounts";
const _LAR = "ListAccountsRequest";
const _LARR = "ListAccountRolesRequest";
const _LARRi = "ListAccountRolesResponse";
const _LARi = "ListAccountsResponse";
const _LARis = "ListAccountRoles";
const _LR = "LogoutRequest";
const _RC = "RoleCredentials";
const _RI = "RoleInfo";
const _RLT = "RoleListType";
const _RNFE = "ResourceNotFoundException";
const _SAKT = "SecretAccessKeyType";
const _STT = "SessionTokenType";
const _TMRE = "TooManyRequestsException";
const _UE = "UnauthorizedException";
const _aI = "accountId";
const _aKI = "accessKeyId";
const _aL = "accountList";
const _aN = "accountName";
const _aT = "accessToken";
const _ai = "account_id";
const _c = "client";
const _e = "error";
const _eA = "emailAddress";
const _ex = "expiration";
const _h = "http";
const _hE = "httpError";
const _hH = "httpHeader";
const _hQ = "httpQuery";
const _m = "message";
const _mR = "maxResults";
const _mr = "max_result";
const _nT = "nextToken";
const _nt = "next_token";
const _rC = "roleCredentials";
const _rL = "roleList";
const _rN = "roleName";
const _rn = "role_name";
const _s = "smithy.ts.sdk.synthetic.com.amazonaws.sso";
const _sAK = "secretAccessKey";
const _sT = "sessionToken";
const _xasbt = "x-amz-sso_bearer_token";
const n0 = "com.amazonaws.sso";
var AccessTokenType = [0, n0, _ATT, 8, 0];
var SecretAccessKeyType = [0, n0, _SAKT, 8, 0];
var SessionTokenType = [0, n0, _STT, 8, 0];
var AccountInfo$ = [3, n0, _AI,
    0,
    [_aI, _aN, _eA],
    [0, 0, 0]
];
var GetRoleCredentialsRequest$ = [3, n0, _GRCR,
    0,
    [_rN, _aI, _aT],
    [[0, { [_hQ]: _rn }], [0, { [_hQ]: _ai }], [() => AccessTokenType, { [_hH]: _xasbt }]], 3
];
var GetRoleCredentialsResponse$ = [3, n0, _GRCRe,
    0,
    [_rC],
    [[() => RoleCredentials$, 0]]
];
var InvalidRequestException$ = [-3, n0, _IRE,
    { [_e]: _c, [_hE]: 400 },
    [_m],
    [0]
];
schema.TypeRegistry.for(n0).registerError(InvalidRequestException$, InvalidRequestException);
var ListAccountRolesRequest$ = [3, n0, _LARR,
    0,
    [_aT, _aI, _nT, _mR],
    [[() => AccessTokenType, { [_hH]: _xasbt }], [0, { [_hQ]: _ai }], [0, { [_hQ]: _nt }], [1, { [_hQ]: _mr }]], 2
];
var ListAccountRolesResponse$ = [3, n0, _LARRi,
    0,
    [_nT, _rL],
    [0, () => RoleListType]
];
var ListAccountsRequest$ = [3, n0, _LAR,
    0,
    [_aT, _nT, _mR],
    [[() => AccessTokenType, { [_hH]: _xasbt }], [0, { [_hQ]: _nt }], [1, { [_hQ]: _mr }]], 1
];
var ListAccountsResponse$ = [3, n0, _LARi,
    0,
    [_nT, _aL],
    [0, () => AccountListType]
];
var LogoutRequest$ = [3, n0, _LR,
    0,
    [_aT],
    [[() => AccessTokenType, { [_hH]: _xasbt }]], 1
];
var ResourceNotFoundException$ = [-3, n0, _RNFE,
    { [_e]: _c, [_hE]: 404 },
    [_m],
    [0]
];
schema.TypeRegistry.for(n0).registerError(ResourceNotFoundException$, ResourceNotFoundException);
var RoleCredentials$ = [3, n0, _RC,
    0,
    [_aKI, _sAK, _sT, _ex],
    [0, [() => SecretAccessKeyType, 0], [() => SessionTokenType, 0], 1]
];
var RoleInfo$ = [3, n0, _RI,
    0,
    [_rN, _aI],
    [0, 0]
];
var TooManyRequestsException$ = [-3, n0, _TMRE,
    { [_e]: _c, [_hE]: 429 },
    [_m],
    [0]
];
schema.TypeRegistry.for(n0).registerError(TooManyRequestsException$, TooManyRequestsException);
var UnauthorizedException$ = [-3, n0, _UE,
    { [_e]: _c, [_hE]: 401 },
    [_m],
    [0]
];
schema.TypeRegistry.for(n0).registerError(UnauthorizedException$, UnauthorizedException);
var __Unit = "unit";
var SSOServiceException$ = [-3, _s, "SSOServiceException", 0, [], []];
schema.TypeRegistry.for(_s).registerError(SSOServiceException$, SSOServiceException);
var AccountListType = [1, n0, _ALT,
    0, () => AccountInfo$
];
var RoleListType = [1, n0, _RLT,
    0, () => RoleInfo$
];
var GetRoleCredentials$ = [9, n0, _GRC,
    { [_h]: ["GET", "/federation/credentials", 200] }, () => GetRoleCredentialsRequest$, () => GetRoleCredentialsResponse$
];
var ListAccountRoles$ = [9, n0, _LARis,
    { [_h]: ["GET", "/assignment/roles", 200] }, () => ListAccountRolesRequest$, () => ListAccountRolesResponse$
];
var ListAccounts$ = [9, n0, _LA,
    { [_h]: ["GET", "/assignment/accounts", 200] }, () => ListAccountsRequest$, () => ListAccountsResponse$
];
var Logout$ = [9, n0, _L,
    { [_h]: ["POST", "/logout", 200] }, () => LogoutRequest$, () => __Unit
];

class GetRoleCredentialsCommand extends smithyClient.Command
    .classBuilder()
    .ep(commonParams)
    .m(function (Command, cs, config, o) {
    return [middlewareEndpoint.getEndpointPlugin(config, Command.getEndpointParameterInstructions())];
})
    .s("SWBPortalService", "GetRoleCredentials", {})
    .n("SSOClient", "GetRoleCredentialsCommand")
    .sc(GetRoleCredentials$)
    .build() {
}

class ListAccountRolesCommand extends smithyClient.Command
    .classBuilder()
    .ep(commonParams)
    .m(function (Command, cs, config, o) {
    return [middlewareEndpoint.getEndpointPlugin(config, Command.getEndpointParameterInstructions())];
})
    .s("SWBPortalService", "ListAccountRoles", {})
    .n("SSOClient", "ListAccountRolesCommand")
    .sc(ListAccountRoles$)
    .build() {
}

class ListAccountsCommand extends smithyClient.Command
    .classBuilder()
    .ep(commonParams)
    .m(function (Command, cs, config, o) {
    return [middlewareEndpoint.getEndpointPlugin(config, Command.getEndpointParameterInstructions())];
})
    .s("SWBPortalService", "ListAccounts", {})
    .n("SSOClient", "ListAccountsCommand")
    .sc(ListAccounts$)
    .build() {
}

class LogoutCommand extends smithyClient.Command
    .classBuilder()
    .ep(commonParams)
    .m(function (Command, cs, config, o) {
    return [middlewareEndpoint.getEndpointPlugin(config, Command.getEndpointParameterInstructions())];
})
    .s("SWBPortalService", "Logout", {})
    .n("SSOClient", "LogoutCommand")
    .sc(Logout$)
    .build() {
}

const commands = {
    GetRoleCredentialsCommand,
    ListAccountRolesCommand,
    ListAccountsCommand,
    LogoutCommand,
};
class SSO extends SSOClient {
}
smithyClient.createAggregatedClient(commands, SSO);

const paginateListAccountRoles = core.createPaginator(SSOClient, ListAccountRolesCommand, "nextToken", "nextToken", "maxResults");

const paginateListAccounts = core.createPaginator(SSOClient, ListAccountsCommand, "nextToken", "nextToken", "maxResults");

Object.defineProperty(exports, "$Command", {
    enumerable: true,
    get: function () { return smithyClient.Command; }
});
Object.defineProperty(exports, "__Client", {
    enumerable: true,
    get: function () { return smithyClient.Client; }
});
exports.AccountInfo$ = AccountInfo$;
exports.GetRoleCredentials$ = GetRoleCredentials$;
exports.GetRoleCredentialsCommand = GetRoleCredentialsCommand;
exports.GetRoleCredentialsRequest$ = GetRoleCredentialsRequest$;
exports.GetRoleCredentialsResponse$ = GetRoleCredentialsResponse$;
exports.InvalidRequestException = InvalidRequestException;
exports.InvalidRequestException$ = InvalidRequestException$;
exports.ListAccountRoles$ = ListAccountRoles$;
exports.ListAccountRolesCommand = ListAccountRolesCommand;
exports.ListAccountRolesRequest$ = ListAccountRolesRequest$;
exports.ListAccountRolesResponse$ = ListAccountRolesResponse$;
exports.ListAccounts$ = ListAccounts$;
exports.ListAccountsCommand = ListAccountsCommand;
exports.ListAccountsRequest$ = ListAccountsRequest$;
exports.ListAccountsResponse$ = ListAccountsResponse$;
exports.Logout$ = Logout$;
exports.LogoutCommand = LogoutCommand;
exports.LogoutRequest$ = LogoutRequest$;
exports.ResourceNotFoundException = ResourceNotFoundException;
exports.ResourceNotFoundException$ = ResourceNotFoundException$;
exports.RoleCredentials$ = RoleCredentials$;
exports.RoleInfo$ = RoleInfo$;
exports.SSO = SSO;
exports.SSOClient = SSOClient;
exports.SSOServiceException = SSOServiceException;
exports.SSOServiceException$ = SSOServiceException$;
exports.TooManyRequestsException = TooManyRequestsException;
exports.TooManyRequestsException$ = TooManyRequestsException$;
exports.UnauthorizedException = UnauthorizedException;
exports.UnauthorizedException$ = UnauthorizedException$;
exports.paginateListAccountRoles = paginateListAccountRoles;
exports.paginateListAccounts = paginateListAccounts;
