from bisect import insort
import itertools
import sys

from common.intervalmap import IntervalMap
from common.misc import im_stream_inter
from sim.SegFreeList import SegFreeList
from sim.TraditionalAllocatorBase import TraditionalAllocatorBase, SegSt

# This is a kind of "throw the kitchen sink at it" approach to dlmalloc; it
# is a merge of DLMStackClusters and DLMWithMemoryTags, so consists, sadly,
# almost entirely of duplicated code.  Barring interactions between the two
# feature sets here, any bugs found here should be fixed upstream.  I wish
# there were a better story.

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
      '_revoke_front', # Put revoked regions at the head of the TIDY list
      '_revoke_jmin',  # Suppress revocation unless this much JUNK accumulated
      '_revoke_tmax',  # Suppress revocation if more than this much TIDY already

      '_prefs' , # va -> Preferred stack flavor
      '_tlpf'  , # TIDY List Per Flavor (flavor -> SegFreeList)
      '_stkfaf', # Stack flavor allocation factor
    )

  @staticmethod
  def _init_add_args(argp) :
    super(__class__, __class__)._init_add_args(argp)
    argp.add_argument('--versions', action='store', type=int, default=16)
    argp.add_argument('--revoke-min', action='store', type=int, default=0,
                      help="Suppress revocations reclaiming fewer JUNK bytes")
    argp.add_argument('--revoke-max-tidy', action='store', type=int, default=None,
                      help="Suppress revocations if sufficient TIDY bytes")
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
    argp.add_argument('--flavor-open-factor', action='store', type=int, default=1024,
                      help="Scale factor when opening a flavored heap region")
    argp.add_argument('--revoke-front', action='store', type=bool, default=True,
                      help="Front revoked spans on the TIDY queue")

  def _init_handle_args(self, args) :
    super(__class__, self)._init_handle_args(args)
    self._nvers = args.versions
    self._revoke_all = args.revoke_all_colors
    self._revoke_jwr = args.revoke_factor
    self._revoke_jmin = args.revoke_min
    self._revoke_tmax = args.revoke_max_tidy
    self._revoke_k = args.revoke_k
    self._revoke_lru = args.revoke_lru
    self._revoke_front = args.revoke_front
    self._stkfaf = args.flavor_open_factor
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
    self._prefs = IntervalMap (
                   self._evp2eva(self._basepg),
                   2**64 - self._evp2eva(self._basepg),
                   None ) 
    self._tlpf = {}
    self._tlpf[None] = SegFreeList()

  def _state_asserts(self) :
    super(__class__,self)._state_asserts()

    #### Stack flavors:

    # For each flavored TIDY list, check that...
    for stk in self._tlpf.keys() :
      # ... The list itself is OK
      self._tlpf[stk].crossreference_asserts()

      # ... and every element ...
      for (pos, sz) in self._tlpf[stk].iterlru() :
        # ... is of the correct flavor
        (pbase, psz, pv) = self._prefs[pos]
        assert pv == stk, ("Mixed-flavor TIDY", (pos, sz, stk), (pbase, psz, pv))
        assert pbase + psz >= pos + sz, ("More flavored TIDY than preferred", (pos, sz, stk), (pbase, psz))

        # ... is actually tidy
        (qbase, qsz, qv) = self._eva2sst[pos]
        assert qv == SegSt.TIDY, ("Flavored TIDY is not TIDY", (pos, sz, stk), (qbase, qsz, qv))
        assert qbase + qsz >= pos + sz, ("More flavor TIDY than TIDY", (pos, sz, stk), (qbase, qsz),
            self._eva2sst[qbase+qsz], list(self._tlpf[stk].iterlru()))

        # ... and really does end where it's supposed to: at the next
        # preference or TIDY boundary
        assert pos + sz == min(pbase + psz, qbase + qsz), \
               ("TIDY segment length mismatch", (pos, sz, stk), (psz, pv), (qsz, qv))

    # Check that all TIDY segments with flavor are on the appropriate flavored TIDY list
    fif = lambda start : ((loc, sz, v) for (loc, sz, v) in self._prefs[start:])
    tif = lambda start : ((loc, sz, v) for (loc, sz, v) in self._eva2sst[start:] if v == SegSt.TIDY)
    for (loc, sz, (stk, _)) in im_stream_inter(fif, tif) :
        fdns = self._tlpf[stk].adns.get(loc,None)
        assert fdns is not None, \
            ("Unindexed flavored TIDY span", (loc, sz, stk),
                self._prefs[loc], self._eva2sst[loc],
                self._tlpf[stk].adns)
        assert (loc, sz) == fdns[0].value, \
            ("Flavored TIDY index mismatch", (loc, sz, stk), fdns[0].value)

    #### MTE:

    # All JUNK spans are at the maximal version, and all non-JUNK spans are
    # at other versions.  Please note that this test is exceptionally slow,
    # due to the large number of segments and color spans that build up, and
    # so it is here but further gated.
    if self._paranoia > 2:
      ist = (x for x in self._eva2sst)
      itg = (x for x in self._mtags)
      for (qb, qsz, (qst, qtv)) in im_stream_inter(lambda _ : ist, lambda _ : itg) :
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
  def _alloc_place_helper(self, stk, sz) :
    #### Stack flavor:

    fit = self._tlpf.setdefault(stk, SegFreeList()).iterfor(sz, 1 << self._alignlog) 

    # Walk the free list to see if we have something of the "stack" flavor laying around.
    for (tpos, tsz) in fit :
        apos = self._eva_align_roundup(tpos)
        if (apos == tpos) or (tpos + tsz >= apos + sz) :
          # Remove from the freelist; any residual spans will come back to us in a moment
          self._tlpf[stk].remove(tpos)
          return apos

    # OK, that didn't work out.  Start over, go claim something without preference bigger
    # than the allocation we're tasked with, repaint it to have the current preference,
    # and return the base thereof.
    #
    # Align to multiple of page sizes
    sz = self._evp2eva(self._eva2evp_roundup(sz * self._stkfaf))

    for (tpos, tsz) in self._tlpf[None].iterfor(sz, 1 << self._alignlog) :
      apos = self._eva_align_roundup(tpos)
      if tpos + tsz >= apos + self._stkfaf * sz :
        self._tlpf[None].remove(tpos)
        self._prefs.mark(tpos, tsz, stk)
        return apos

    # OK, OK, we really haven't found anything, even if we're willing to
    # repaint.  Go grab at the wilderness; bump the wilderness pointer now to trigger
    # the common case in _mark_allocated; things will be enqueued on our per-flavor TIDY
    # list using _mark_allocated_residual
    #
    # XXX? Round base up to page boundaries and allocate the whole
    # thing for this flavor.

    # pos = self._eva_align_roundup(self._wildern)
    pos = self._evp2eva(self._eva2evp_roundup(self._wildern))
    self._wildern = pos + sz
    self._prefs.mark(pos, self._wildern - pos, stk)

    # This is kind of gross; this segment is not in any of our flavored free
    # lists, and we don't want to put it there, as our _mark_tidy would do.
    # So instead, we reach up to the superclass to update the segment state
    # and rely on duplicating any other work that our _mark_tidy does
    # (which, thankfully, is currently none)
    super()._mark_tidy(pos, sz)

    return pos
  def _alloc_place(self, stk, sz) :
    pos = self._alloc_place_helper(stk, sz)

    #### MTE:

    nv = max(v for (_, __, v) in self._mtags[pos:pos+sz])
    assert nv != self._nvers
    self._mtags.mark(pos, sz, nv)

    return pos

  def _mark_allocated_residual(self, stk, loc, sz, isLeft):
    super()._mark_allocated_residual(stk, loc, sz, isLeft)

    #### Stack flavor:

    # This is a bit of a mess.  The "residual" spans that come back to us
    # are from the generic TIDY list, which coalesces across our preferences.
    # So, only queue the residual bits to the preference-respecting TIDY list
    # if the current allocation stack matches the span's preference, and, only then,
    # up to our preference's boundary.  Discontiguous stk-flavored TIDY spans will
    # already be in the right free list (proof by induction).
    if isLeft :
      (qbase, qsz, qv) = self._prefs[loc+sz-1]
      if stk == qv :
        base = max(qbase, loc)
        self._tlpf[stk].expunge(base, loc + sz - base)
        self._tlpf[stk].insert(base, loc + sz - base)
    else :
      (qbase, qsz, qv) = self._prefs[loc]
      if stk == qv :
        lim = min(qbase + qsz, loc + sz)
        self._tlpf[stk].expunge(loc, lim - loc)
        self._tlpf[stk].insert(loc, lim - loc)

  def _free(self, stk, tid, loc) :
    sz = self._eva2sz[loc]

    (_, __, v) = self._mtags.get(loc)
    if v == self._nvers-1 :
      super(__class__,self)._free(stk, tid, loc)
      self._mtags.mark(loc, sz, self._nvers)
    else :
      super(__class__,self)._free_unsafe(stk, tid, loc)
      self._mtags.mark(loc, sz, v+1)

  def _mark_tidy(self, loc, sz) :
    super()._mark_tidy(loc, sz)

    #### Stack flavor:

    for (tloc, tsz, tv) in self._prefs[loc:loc+sz] :
      nloc = max(tloc, loc)
      nsz = min(loc + sz, tloc + tsz) - nloc
      self._tlpf[tv].insert(nloc, nsz)

  def _mark_revoked(self, loc, sz):
    #### Stack flavor:

    # XXX Should we be preserving preferences?  Always?  Sometimes?

    # Remove each overlapping span from each preferrential TIDY list; the parent allocator
    # will take care of managing the global TIDY list.
    for (_, __, pv) in self._prefs[loc:loc+sz] :
      self._tlpf[pv].expunge(loc, sz)

    if sz >= 0 : # XXX
      # Just paint the whole thing as None.
      self._tlpf[None].insert(loc, sz, front=self._revoke_front)
      self._prefs.mark(loc, sz, None)
    else :
      # Preserve preferences and re-queue
      for (qb, qsz, pv) in self._prefs[loc:loc+sz] :
        b = max(qb, loc)
        l = min(qb + qsz, loc + sz)
        self._tlpf[pv].insert(b, l-b, front=self._revoke_front)

    #### MTE:
    self._mtags.mark(loc, sz, 0)

    super()._mark_tidy(loc,sz)

  # Sort by the sum of (sz*version), as that is the value by which we wind
  # back the clock to defer later revocations.
  def _find_clockiest_revokable_spans(self, n=1):
    if n == 0 : return
    if n == 1 and self._brscache is not None :
        return [self._brscache]

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

  def _revoke_iterator(self):
    for (b,s,v) in self._eva2sst.iter_vfilter(None, self._wildern,
                            sst_tj if self._revoke_all else [SegSt.JUNK]) :
      jsum = sum(sz for (_, sz, v) in self._eva2sst[b:b+s] if v == SegSt.JUNK)
      if jsum == 0 : continue
      yield (jsum, b, s)

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

    # Not enough JUNK
    if self._njunk < self._revoke_jmin : return

    # Still enough TIDY?
    if self._revoke_tmax is not None and \
       self._wildern - self._nwait \
         - self._njunk - self.evp2eva(self._baseva) \
        > self._revoke_tmax :
      return

    if self._revoke_k is None :
      it = self._revoke_iterator()
    else:
      # Allocate room for up to _revoke_lru things from the JUNK LRU.
      nlru = min(self._revoke_lru, len(self._junklru))

      # XXX I'd love to, but boy is this a slow way to do it.
      #
      ## # Estimate reclaim quantity
      ## unjunk = self._find_largest_revokable_spans(n=self._revoke_k-nlru)
      ## unjunk = sum(nj for (nj, _, __) in unjunk)
      ## if unjunk < self._revoke_jmin:
      ##   return
      #
      # This is going to be much faster to simulate due to the brscache, though
      # it overestimates the amout we will reclaim from non-LRU spans and gives
      # no credit to LRU spans.
      #
      unjunk = self._find_largest_revokable_spans(n=1)
      if len(unjunk) * (self._revoke_k - nlru) < self._revoke_jmin :
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

    rl = list(it)
    rl.reverse() # XXX reverse is a hack (lower VA last, fronted on LRU queues?)
    self._do_revoke(rl)
    if __debug__: self._state_asserts()
