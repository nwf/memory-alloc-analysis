from bisect import insort
import itertools
import sys

from common.intervalmap import IntervalMap
from common.misc import im_stream_inter
from sim.TraditionalAllocatorBase import TraditionalAllocatorBase, SegSt

# What if we have memory versioning?  That is, ceil(log2(N)) bits per every
# aligned K bytes of memory and in every capability, with enforced version
# matching on dereference (at least).
#
# We repurpose the existing --align-log functionality to allow varying K.
#
# TIDY and JUNK blur together a bit:
#
#  Upon revocation, memory shall have version 0 and no pointers to it (TIDY)
#
#  Memory is exhausted and must be revoked when it is freed and its version
#  is N-2 prior to free; its version will be N-1 after free (JUNK).  All
#  these spans are on the JUNK queue.
#
#  In the middle, we have allocable spans whose versions are greater than
#  any pointer given out to them.  We keep these on the TIDY queue, which is
#  now the "sufficiently TIDY" queue, or somesuch.  Placing here increments
#  the version tags on memory.
#
# We choose to never grant an allocation with version N-1.  While we could
# do so, we would have to not increment the version on free.  This
# fence-post disposition gives a "prompt detection" of UAF regardless of
# version. The other options give prompt detection only at other versions or
# only after reallocation.

sst_tj  = { SegSt.TIDY, SegSt.JUNK }

class Allocator(TraditionalAllocatorBase):

  __slots__ = (
      '_mtags' # Memory tag version intervalmap
      '_nvers', # Number of versions
      '_revoke_all', # Concurrency is infinitely fast; revoke all free spans
      '_revoke_jwr', # JUNK/WAIT for revocation
      '_revoke_k', # Limited revocation facilities
      '_revoke_lru', # With revoke_k, sample from junklru, too.
    )

  @staticmethod
  def _init_add_args(argp) :
    super(__class__, __class__)._init_add_args(argp)
    argp.add_argument('--versions', action='store', type=int, default=16)
    argp.add_argument('--revoke-min', action='store', type=int, default=0,
                      help="Suppress revocations reclaiming fewer JUNK bytes")
    argp.add_argument('--revoke-factor', action='store', type=float, default=None,
                      help="Ratio of JUNK to WAIT triggering revocation")
    argp.add_argument('--revoke-k', action='store', type=int, default=None,
                      help="Assume limited revocation facilities")
    argp.add_argument('--revoke-lru', action='store', type=int, default=None,
                      help="Ensure old maximal-versioned spans eventually recycled")
    argp.add_argument('--revoke-sort', action='store', type=str,
                      default="clock", choices=["clock", "size"],
                      help="Selection function for limited revocation")
    argp.add_argument('--revoke-all-colors', action='store_true', default=False,
                      help="Revoke all free spans, not just maximal-versioned")

  def _init_handle_args(self, args) :
    super(__class__, self)._init_handle_args(args)
    self._nvers = args.versions
    self._revoke_all = args.revoke_all_colors
    self._revoke_jwr = args.revoke_factor
    self._revoke_min = args.revoke_min
    self._revoke_k = args.revoke_k
    self._revoke_lru = args.revoke_lru
    if self._revoke_lru is not None :
      if self._revoke_k is None : self._revoke_k = self._revoke_lru
      else                      : assert self._revoke_k >= self._revoke_lru
    else :
      self._revoke_lru = 0

    if   args.revoke_sort == "clock" :
      self._find_largest_revokable_spans = self._find_clockiest_revokable_spans
    elif args.revoke_sort == "size"  :
      pass

    if args.revoke_k is None and args.revoke_lru is not None:
      raise ValueError("--revoke-lru only sensible with --revoke-k")

    if args.revoke_factor is None and args.revoke_min == 0 :
      raise ValueError("Please restrain revocation with --revoke-factor or --revoke-min")

  def __init__(self, *args, **kwargs) :
    super(__class__,self).__init__(*args, **kwargs)

    self._mtags = IntervalMap(
      self._evp2eva(self._basepg),
      2**64 - self._evp2eva(self._basepg),
      0
    )

  def _state_asserts(self) :
    super(__class__,self)._state_asserts()

    # All JUNK spans are at the maximal version, and all non-JUNK spans are
    # at other versions.
    ist = (x for x in self._eva2sst)
    itg = (x for x in self._mtags)
    for (qb, qsz, qvs) in im_stream_inter(lambda _ : ist, lambda _ : itg) :
      (qst, qtv) = qvs
      if   qst == SegSt.JUNK : assert qtv == self._nvers, (qb, qsz, qtv)
      elif qst == SegSt.AHWM : assert qtv == 0
      else                   : assert qtv != self._nvers

  # XXX
  # We'd like to ask a slightly different question, namely: where can we
  # place this to minimize the advancement of the version clocks.  You might
  # think we'd never advance in allocate, having advanced in free, but if we
  # choose a place formed from several freed spans, we have to advance to the
  # max of all spans we end up using, which might advance some of the clocks
  # quite a bit.
  # ...
  # At the moment, tidylst coalesces all versions together.  We should
  # instead hunt for a minimum of the (byte*clock_delta) sum for the places
  # considered.  We can stop early if we find a zero, of course.
  #
  def _alloc_place(self, stk, sz) :
    pos = super()._alloc_place(stk, sz)
    nv = max(v for (_, __, v) in self._mtags[pos:pos+sz])
    assert nv != self._nvers
    self._mtags.mark(pos, sz, nv)
    return pos

  def _free(self, stk, tid, loc) :
    sz = self._eva2sz[loc]
    (_, __, v) = self._mtags.get(loc)
    if v == self._nvers-1 :
      super(__class__,self)._free(stk, tid, loc)
      self._mtags.mark(loc, sz, self._nvers)
    else :
      super(__class__,self)._free_unsafe(stk, tid, loc)
      self._mtags.mark(loc, sz, v+1)

  def _mark_revoked(self, loc, sz):
    super()._mark_revoked(loc,sz)
    self._mtags.mark(loc, sz, 0)

  # Sort by the sum of (sz*version), as that is the value by which we wind
  # back the clock to defer later revocations.
  def _find_clockiest_revokable_spans(self, n=1):
    if n == 0 : return

    bests = [(0, 0, -1, -1)] # [(clocksum, njunk, loc, sz)] in ascending order
    for (qbase, qsz, qv) in self._eva2sst.iter_vfilter(None, self._wildern, sst_tj) :

        clocksum = 0
        for (vbase, vsz, vv) in self._mtags[qbase:qbase+qsz] :
          # Don't walk off the end of the last segment
          vsz = min(vsz, qbase + qsz - vbase)
          clocksum += vsz * vv

        if clocksum <= bests[0][0] : continue

        # For internal accounting, also accumulate the number of JUNK bytes
        nj = sum([sz for (_, sz, v) in self._eva2sst[qbase:qbase+qsz]
                  if v == SegSt.JUNK])
        insort(bests, (clocksum, nj, qbase, qsz))

        bests = bests[(-n):]

    return [best[1:] for best in bests if best[2] >= 0]

  # Revocation here does not have a fixed number of windows on which it can
  # operate; everything in the junk (and tidy!) lists can be revoked in a
  # single go.  In implementation, this looks like validating that every
  # capability's contained version field matches the version painted in RAM.
  #
  # The limits of the allocator behavior range from being able to actually
  # get all free spans, regardless of their version (JUNK or TIDY), to being
  # able to reclaim just the ones that were already at the maximal version
  # (i.e. JUNK).  In practice, one could imagine the allocator maintaining
  # a "free epoch" bit and reclaiming (i.e., restoring to version 0) all
  # segments whose free epoch predates the current, now-ending sweep.
  #
  def _maybe_revoke(self):
    # Not above ratio threshold
    if self._revoke_jwr is not None \
       and self._njunk < self._revoke_jwr * self._nwait:
        return

    if self._njunk < self._revoke_min :
      return

    if self._revoke_k is None :
      it = ((jsz, jb, jsz) for (jb, jsz) in self._junklru)
      if self._revoke_all :
        it = itertools.chain(it, ((0, tb, tsz) for (tb, tsz) in self._tidylst.iterlru()))
    else:
      # Allocate room for up to _revoke_lru things from the JUNK LRU.
      nlru = min(self._revoke_lru, len(self._junklru))

      # XXX I'd love to, but boy is this a slow way to do it.
      #
      ## # Estimate reclaim quantity
      ## unjunk = self._find_largest_revokable_spans(n=self._revoke_k-nlru)
      ## unjunk = sum(nj for (nj, _, __) in unjunk)
      ## if unjunk < self._revoke_min:
      ##   return
      #
      # This is going to be much faster due to the brscache.
      #
      unjunk = self._find_largest_revokable_spans(n=1)
      if len(unjunk) * self._revoke_k < self._revoke_min :
        return

      # Do that again, but as part of the set of things to push to the revoker
      it = (x for x in self._find_largest_revokable_spans(n=self._revoke_k-nlru))

      # Add from the LRU JUNK queue
      it = itertools.chain(it, ((jsz, jb, jsz) for (jb, jsz) in self._junklru))

      # XXX Could also pull from the TIDY queue, but should filter by those
      # that contain nonzero versions or sort by the clockiest span or
      # something.
      #
      # XXX Should also coalesce with TIDY spans for the things we pull from
      # the LRU queue.

      # Limit to the number we actually can run
      # XXX This should deduplicate before slicing.  Sigh.
      #
      it = itertools.islice(it, self._revoke_k)

    self._do_revoke(list(it))
