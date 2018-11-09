import itertools
from sim.TraditionalAllocatorBase import TraditionalAllocatorBase, PageSt

class Allocator(TraditionalAllocatorBase):

  __slots__ = (
    '_revoke_k', # Number of regions to simultaneously revoke
    '_revoke_jmin',  # Suppress revocation unless this much JUNK accumulated
  )

  @staticmethod
  def _init_add_args(argp):
    super(__class__, __class__)._init_add_args(argp)
    argp.add_argument('--revoke-k', action='store', type=int, default=16)
    argp.add_argument('--revoke-min', action='store', type=int, default=0,
                      help="Suppress revocations reclaiming fewer JUNK bytes")
    argp.add_argument('--unsafe-reuse', action='store_const', const=True,
                      default=False,
                      help='free immediately to reusable state')

  def _init_handle_args(self, args) :
    super(__class__, self)._init_handle_args(args)

    if args.unsafe_reuse :
      self._free = self._free_unsafe

    assert args.revoke_k > 0
    self._revoke_k        = args.revoke_k

    self._revoke_jmin = args.revoke_min

  def _maybe_revoke(self):
    # XXX configurable policy

    # Not enough JUNK?
    if self._njunk < self._revoke_jmin : return

    # if self._njunk >= self._nwait and len(self._junklru) >= 16 :
    #   self._do_revoke_best_and(revoke=[loc for (loc, _) in itertools.islice(self._junklru,8)])

    # If we are over AS budget, always revoke a mixture of largest and
    # oldest spans, including unmapped junk.
    if self._njunk >= self._nwait:
      self._do_revoke_best_and(revoke=[loc for (loc, _) in itertools.islice(self._junklru,8)])
      return

    # If we are not over AS budget, find the oldest mapped junk and revoke
    # those so we can reuse them, trying to minimize our count of junk but
    # mapped regions.
    komj = list(itertools.islice(
            ((jb, jsz, jb) for (jb, jsz, pst) in
                                ((jb, jsz, self._evp2pst.get(self._eva2evp(jb)))
                                for (jb, jsz) in self._junklru)
                          if pst == PageSt.MAPD),
            self._revoke_k))
    if len(self._junklru) >= 16 and komj != []:
      self._do_revoke(komj)
