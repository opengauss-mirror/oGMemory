const IDENTITY_TTL_MS = 2 * 60 * 60 * 1000;
const MAX_ALIASES = 2_048;
const text = (value) => typeof value === "string" ? value.trim() : "";
const aliasesFor = (params, context) => [...new Set([
  params?.sessionId, context?.sessionId, params?.sessionKey, context?.sessionKey,
].map(text).filter(Boolean))];

export function createSessionAgentResolver(defaultAgentId, now = Date.now) {
  const identities = new Map();
  const retained = new Map();
  let revision = 0;
  return {
    resolve(params = {}, context = {}) {
      const timestamp = now();
      for (const [alias, identity] of identities) {
        if (timestamp - identity.seen > IDENTITY_TTL_MS) identities.delete(alias);
      }
      const aliases = aliasesFor(params, context);
      const matches = new Set(aliases.flatMap((alias) => [identities.get(alias), retained.get(alias)]).filter(Boolean));
      let identity;
      for (const candidate of matches) {
        if (!identity || Number(candidate.explicit) > Number(identity.explicit)
          || (candidate.explicit === identity.explicit && candidate.revision > identity.revision)) {
          identity = candidate;
        }
      }
      identity ||= {};
      const sessionAgent = [params?.sessionKey, context?.sessionKey]
        .map((key) => /^agent:([^:]+):/.exec(text(key))?.[1]).find(Boolean);
      const explicitAgent = text(params?.agentId) || text(context?.agentId);
      const agentId = explicitAgent || identity.agentId || sessionAgent;
      if (agentId) {
        if (explicitAgent || !identity.agentId) {
          identity.explicit = Boolean(explicitAgent);
          identity.revision = ++revision;
        }
        identity.agentId = agentId;
        identity.seen = timestamp;
        const refreshedAliases = new Set(aliases);
        for (const [alias, previous] of identities) {
          if (matches.has(previous)) refreshedAliases.add(alias);
        }
        for (const alias of refreshedAliases) {
          identities.delete(alias);
          identities.set(alias, identity);
        }
        let keepIdentity = false;
        for (const [alias, previous] of retained) {
          if (matches.has(previous) || refreshedAliases.has(alias)) {
            retained.set(alias, identity);
            keepIdentity = true;
          }
        }
        if (keepIdentity) {
          for (const alias of refreshedAliases) retained.set(alias, identity);
        }
        while (identities.size > MAX_ALIASES) identities.delete(identities.keys().next().value);
      }
      return agentId || defaultAgentId;
    },
    retain(sessionId) {
      const alias = text(sessionId);
      if (!alias) return;
      const identity = identities.get(alias) || retained.get(alias);
      retained.set(alias, identity);
      if (identity) {
        for (const [key, candidate] of identities) {
          if (candidate === identity) retained.set(key, identity);
        }
      }
    },
    forget(params = {}, context = {}) {
      const aliases = aliasesFor(params, context);
      const matches = new Set(aliases.flatMap((alias) => [identities.get(alias), retained.get(alias)]).filter(Boolean));
      for (const [alias, identity] of identities) {
        if (matches.has(identity)) identities.delete(alias);
      }
      for (const [alias, identity] of retained) {
        if (matches.has(identity) || aliases.includes(alias)) retained.delete(alias);
      }
    },
  };
}
