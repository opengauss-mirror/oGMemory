const TAKEOVER_COMPACTION_BOUNDARY_CUSTOM_TYPE =
  "og-memory:takeover-compaction-boundary";

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function getBranch(sessionManager) {
  if (typeof sessionManager?.getBranch !== "function") {
    return [];
  }
  const branch = sessionManager.getBranch();
  return Array.isArray(branch) ? branch : [];
}

function getEntryId(entry) {
  return normalizeString(entry?.id);
}

function getMessageRole(entry) {
  return entry?.type === "message" ? normalizeString(entry.message?.role) : "";
}

function isToolResultMessage(message) {
  const role = normalizeString(message?.role);
  const type = normalizeString(message?.type);
  return role === "toolResult" || role === "tool" || type === "toolResult";
}

function blockType(block) {
  return normalizeString(block?.type).toLowerCase();
}

function messageHasToolCall(message) {
  const role = normalizeString(message?.role);
  if (role !== "assistant") {
    return false;
  }
  if (
    (Array.isArray(message?.toolCalls) && message.toolCalls.length > 0) ||
    (Array.isArray(message?.tool_calls) && message.tool_calls.length > 0) ||
    (Array.isArray(message?.functionCall) && message.functionCall.length > 0) ||
    (Array.isArray(message?.function_call) && message.function_call.length > 0)
  ) {
    return true;
  }
  const objectToolCall =
    (message?.functionCall &&
      typeof message.functionCall === "object" &&
      !Array.isArray(message.functionCall)) ||
    (message?.function_call &&
      typeof message.function_call === "object" &&
      !Array.isArray(message.function_call));
  if (objectToolCall) {
    return true;
  }
  const stopReason = normalizeString(message?.stopReason).toLowerCase();
  if (["tool_calls", "tooluse", "tool_use", "function_call"].includes(stopReason)) {
    return true;
  }
  const content = message?.content;
  const blocks = Array.isArray(content)
    ? content
    : content && typeof content === "object"
      ? [content]
      : [];
  return blocks.some((block) =>
    [
      "function_call",
      "function_calls",
      "functioncall",
      "functioncalls",
      "tool_use",
      "toolcall",
      "tool_call",
      "tooluse",
    ].includes(blockType(block)),
  );
}

function tailContainsUnsafeToolBlock(branch, startIndex) {
  for (let i = Math.max(0, startIndex); i < branch.length; i += 1) {
    const entry = branch[i];
    if (entry?.type !== "message") {
      continue;
    }
    const message = entry.message;
    if (isToolResultMessage(message) || messageHasToolCall(message)) {
      return true;
    }
  }
  return false;
}

function findBranchIndexById(branch, id) {
  const normalizedId = normalizeString(id);
  if (!normalizedId) {
    return -1;
  }
  return branch.findIndex((entry) => getEntryId(entry) === normalizedId);
}

function isSafeBoundaryId(branch, id) {
  const index = findBranchIndexById(branch, id);
  return index >= 0 && !tailContainsUnsafeToolBlock(branch, index);
}

function findLatestSafeTailBoundary(branch) {
  for (const preferredRole of ["user", "assistant"]) {
    for (let i = branch.length - 1; i >= 0; i -= 1) {
      const entry = branch[i];
      if (entry?.type !== "message" || getMessageRole(entry) !== preferredRole) {
        continue;
      }
      if (!tailContainsUnsafeToolBlock(branch, i)) {
        return getEntryId(entry);
      }
    }
  }

  for (let i = branch.length - 1; i >= 0; i -= 1) {
    const entry = branch[i];
    if (entry?.type === "message" && !tailContainsUnsafeToolBlock(branch, i)) {
      return getEntryId(entry);
    }
  }
  return "";
}

function isManualCompaction(params) {
  const trigger = normalizeString(params?.runtimeContext?.trigger);
  return (
    trigger === "manual" ||
    (params?.force === true && params?.compactionTarget === "threshold")
  );
}

function chooseTakeoverCompactionBoundary({
  branch,
  currentLeafId,
  upstreamFirstKeptEntryId,
  params,
}) {
  const upstream = normalizeString(upstreamFirstKeptEntryId);
  const safeTail = findLatestSafeTailBoundary(branch);
  const firstBranchId = getEntryId(branch[0]);

  if (branch.length === 0) {
    return {
      firstKeptEntryId: normalizeString(currentLeafId),
      hardCheckpoint: false,
      reason: "leaf_fallback",
    };
  }

  if (isManualCompaction(params)) {
    return {
      firstKeptEntryId: safeTail || normalizeString(currentLeafId),
      hardCheckpoint: true,
      reason: "manual",
    };
  }

  if (upstream && findBranchIndexById(branch, upstream) >= 0) {
    return {
      firstKeptEntryId: upstream,
      hardCheckpoint: false,
      reason: "upstream",
    };
  }

  if (safeTail) {
    return {
      firstKeptEntryId: safeTail,
      hardCheckpoint: false,
      reason: "safe_tail",
    };
  }

  return {
    firstKeptEntryId: firstBranchId || normalizeString(currentLeafId),
    hardCheckpoint: true,
    reason: "no_safe_tail",
  };
}

function appendTakeoverBoundaryMarker(sessionManager, data) {
  if (typeof sessionManager?.appendCustomEntry !== "function") {
    return "";
  }
  const returnedId = normalizeString(
    sessionManager.appendCustomEntry(TAKEOVER_COMPACTION_BOUNDARY_CUSTOM_TYPE, data),
  );
  if (returnedId) {
    return returnedId;
  }
  const leafEntry =
    typeof sessionManager.getLeafEntry === "function" ? sessionManager.getLeafEntry() : null;
  if (
    leafEntry?.type === "custom" &&
    leafEntry.customType === TAKEOVER_COMPACTION_BOUNDARY_CUSTOM_TYPE
  ) {
    return getEntryId(leafEntry);
  }
  return "";
}

function buildTakeoverBoundaryMarkerData({
  reason,
  currentLeafId,
  selectedFirstKeptEntryId,
  upstreamFirstKeptEntryId,
}) {
  return {
    kind: "takeover-compaction-boundary",
    reason: normalizeString(reason),
    previousLeafId: normalizeString(currentLeafId),
    selectedFirstKeptEntryId: normalizeString(selectedFirstKeptEntryId),
    upstreamFirstKeptEntryId: normalizeString(upstreamFirstKeptEntryId),
    createdAt: new Date().toISOString(),
  };
}

function appendHardCheckpoint({
  sessionManager,
  summary,
  tokensBefore,
  details,
  boundary,
  currentLeafId,
  upstreamFirstKeptEntryId,
}) {
  const markerEntryId = appendTakeoverBoundaryMarker(
    sessionManager,
    buildTakeoverBoundaryMarkerData({
      reason: boundary.reason,
      currentLeafId,
      selectedFirstKeptEntryId: boundary.firstKeptEntryId,
      upstreamFirstKeptEntryId,
    }),
  );
  if (markerEntryId) {
    const compactionEntryId = sessionManager.appendCompaction(
      summary,
      markerEntryId,
      tokensBefore,
      details,
    );
    return {
      firstKeptEntryId: markerEntryId,
      compactionEntryId,
      reused: false,
    };
  }

  const anchorCompactionEntryId = sessionManager.appendCompaction(
    summary,
    boundary.firstKeptEntryId,
    tokensBefore,
    details,
  );
  const finalCompactionEntryId = sessionManager.appendCompaction(
    summary,
    anchorCompactionEntryId,
    tokensBefore,
    details,
  );
  return {
    firstKeptEntryId: anchorCompactionEntryId,
    compactionEntryId: finalCompactionEntryId,
    reused: false,
  };
}

export async function appendTakeoverCompactionBoundary({
  sessionFile,
  summary,
  tokensBefore,
  details,
  params,
  upstreamFirstKeptEntryId,
  loadSessionManagerModule = async () => import("@mariozechner/pi-coding-agent"),
} = {}) {
  const normalizedSessionFile =
    typeof sessionFile === "string" ? sessionFile.trim() : "";
  const normalizedSummary = typeof summary === "string" ? summary.trim() : "";
  const normalizedTokensBefore =
    Number.isFinite(tokensBefore) && tokensBefore >= 0 ? Math.floor(tokensBefore) : null;

  if (!normalizedSessionFile || normalizedTokensBefore === null) {
    return null;
  }

  try {
    const mod = await loadSessionManagerModule();
    const SessionManager = mod?.SessionManager;
    if (!SessionManager || typeof SessionManager.open !== "function") {
      return null;
    }

    const sessionManager = SessionManager.open(normalizedSessionFile);
    if (typeof sessionManager.appendCompaction !== "function") {
      return null;
    }

    const leafEntry =
      typeof sessionManager.getLeafEntry === "function" ? sessionManager.getLeafEntry() : null;
    const leafId =
      typeof sessionManager.getLeafId === "function"
        ? String(sessionManager.getLeafId() ?? "").trim()
        : "";
    const currentLeafId =
      leafId || (leafEntry && typeof leafEntry.id === "string" ? leafEntry.id.trim() : "");
    const branch = getBranch(sessionManager);

    if (!currentLeafId) {
      return null;
    }

    if (leafEntry?.type === "compaction") {
      const existingFirstKeptEntryId =
        typeof leafEntry.firstKeptEntryId === "string" && leafEntry.firstKeptEntryId.trim()
          ? leafEntry.firstKeptEntryId.trim()
          : currentLeafId;
      if (isManualCompaction(params) || !isSafeBoundaryId(branch, existingFirstKeptEntryId)) {
        return appendHardCheckpoint({
          sessionManager,
          summary: normalizedSummary,
          tokensBefore: normalizedTokensBefore,
          details,
          boundary: {
            firstKeptEntryId: existingFirstKeptEntryId,
            reason: isManualCompaction(params)
              ? "manual_existing_compaction"
              : "unsafe_existing_compaction",
          },
          currentLeafId,
          upstreamFirstKeptEntryId,
        });
      }
      return {
        firstKeptEntryId: existingFirstKeptEntryId,
        compactionEntryId: leafEntry.id,
        reused: true,
      };
    }

    const boundary = chooseTakeoverCompactionBoundary({
      branch,
      currentLeafId,
      upstreamFirstKeptEntryId,
      params,
    });

    if (boundary.hardCheckpoint) {
      return appendHardCheckpoint({
        sessionManager,
        summary: normalizedSummary,
        tokensBefore: normalizedTokensBefore,
        details,
        boundary,
        currentLeafId,
        upstreamFirstKeptEntryId,
      });
    }

    const compactionEntryId = sessionManager.appendCompaction(
      normalizedSummary,
      boundary.firstKeptEntryId,
      normalizedTokensBefore,
      details,
    );
    return {
      firstKeptEntryId: boundary.firstKeptEntryId,
      compactionEntryId,
      reused: false,
    };
  } catch (err) {
    console.warn(
      `[og-memory] failed to persist takeover compaction boundary: ${String(err)}`,
    );
    return null;
  }
}
