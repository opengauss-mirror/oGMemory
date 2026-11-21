from collections import Counter, OrderedDict
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from core.models import RequestContext


@dataclass(frozen=True)
class _Turn:
    ctx: RequestContext
    ready_at_ms: int

    @property
    def key(self):
        return self.ctx.account_id, self.ctx.user_id, self.ctx.agent_id, self.ctx.session_id


class AgentTurnQueue:
    """Session heads coordinated under the retry worker's condition lock."""

    def __init__(self):
        self._ready = OrderedDict()
        self._delayed = []
        self._known = set()
        self._known_accounts = Counter()
        self._active = {}
        self._sequence = count()
        self._last_account = None
        self._last_user = {}

    def wake(self, ctx, ready_at_ms, due_at, now, *, pending=True):
        turn = _Turn(ctx, ready_at_ms)
        if turn.key in self._known:
            if pending and turn.key in self._active and self._active[turn.key] is None:
                self._active[turn.key] = turn
            return
        self._known.add(turn.key)
        self._known_accounts[ctx.account_id] += 1
        self._put(turn, due_at, now)

    def _put(self, turn, due_at, now):
        if due_at > now:
            heappush(self._delayed, (due_at, next(self._sequence), turn))
        else:
            users = self._ready.setdefault(turn.ctx.account_id, OrderedDict())
            sessions = users.setdefault(turn.ctx.user_id, [])
            heappush(sessions, (turn.ready_at_ms, next(self._sequence), turn))

    def pop(self, now):
        while self._delayed and self._delayed[0][0] <= now:
            due_at, _, turn = heappop(self._delayed)
            self._put(turn, due_at, now)
        if not self._ready:
            return None
        account, users = self._next_group(self._ready, self._last_account)
        user, sessions = self._next_group(users, self._last_user.get(account))
        _, _, turn = heappop(sessions)
        if not sessions:
            del users[user]
        self._last_account = account
        self._last_user[account] = user
        if not users:
            del self._ready[account]
        self._active[turn.key] = None
        return turn

    @staticmethod
    def _next_group(groups, last):
        first = next(iter(groups))
        if len(groups) > 1 and first == last:
            groups.move_to_end(first)
        first = next(iter(groups))
        groups.move_to_end(first)
        return first, groups[first]

    def next_deadline(self):
        return self._delayed[0][0] if self._delayed else None

    def finish(self, turn, delay, now, ready_at_ms):
        wake = self._active.pop(turn.key)
        if delay is None and wake is None:
            self._known.remove(turn.key)
            account = turn.ctx.account_id
            self._known_accounts[account] -= 1
            if not self._known_accounts[account]:
                del self._known_accounts[account]
                self._last_user.pop(account, None)
            return
        if not delay:
            turn = wake or _Turn(turn.ctx, ready_at_ms)
        self._put(turn, now + (delay or 0), now)
