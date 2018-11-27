import itertools

from common.intervalmap import IntervalMap
from common.misc import im_stream_inter
from sim.SegFreeList import SegFreeList
from sim.TraditionalAllocatorBase import TraditionalAllocatorBase, SegSt

class Allocator(TraditionalAllocatorBase):

  slots = (
      '_prefs' , # Preferred stack flavor for regions of memory
      '_tlpf'  , # TIDY List Per Flavor
      '_stkfaf', # Stack flavor allocation factor
  )

  # The parent class's _tidylst will continue to be all TIDY spans of any
  # flavor, but we additionally track TIDY-per-flavor in _tlpf, including
  # the unflavored spans at _tlpf[None]

  @staticmethod
  def _init_add_args(argp) :
    super(__class__, __class__)._init_add_args(argp)
    argp.add_argument('--flavor-open-factor', action='store', type=int, default=1024,
                      help="Scale factor when opening a flavored heap region")

  def _init_handle_args(self, args) :
    super(__class__, self)._init_handle_args(args)
    self._stkfaf = args.flavor_open_factor

  def __init__(self, **kwargs) :
    super().__init__(**kwargs)

    # XXX
    self._revoke_k = 8
    self._free = self._free_unsafe

    self._prefs = IntervalMap (
                   self._evp2eva(self._basepg),
                   2**64 - self._evp2eva(self._basepg),
                   None ) 
    self._tlpf = {}
    self._tlpf[None] = SegFreeList()

  def _state_asserts(self):
    super()._state_asserts()

    # For each flavored TIDY list, check that...
    for stk in self._tlpf.keys() :
      # ... The list itself is OK
      self._tlpf[stk].crossreference_asserts()

      # ... and every element ...
      for (pos, sz) in self._tlpf[stk].iterlru() :
        # ... is of the correct flavor
        (qbase, qsz, qv) = self._prefs[pos]
        assert qv == stk, ("Mixed-flavor TIDY", (pos, sz, stk), (qbase, qsz, qv))
        assert qbase + qsz >= pos + sz, ("More preferred TIDY than flavored", (pos, sz, stk), (qbase, qsz))

        # ... is actually tidy
        (qbase, qsz, qv) = self._eva2sst[pos]
        assert qv == SegSt.TIDY, ("Flavored TIDY is not TIDY", (pos, sz, stk), (qbase, qsz, qv))
        assert qbase + qsz >= pos + sz, ("More flavor TIDY than TIDY", (pos, sz, stk), (qbase, qsz),
            self._eva2sst[qbase+qsz], list(self._tlpf[stk].iterlru()))

    # Check that all TIDY segments with flavor are on the appropriate flavored TIDY list
    fif = lambda start : ((loc, sz, v) for (loc, sz, v) in self._prefs[start:])
    tif = lambda start : ((loc, sz, v) for (loc, sz, v) in self._eva2sst[start:] if v == SegSt.TIDY)
    for (loc, sz, (stk, _)) in im_stream_inter(fif, tif) :
        assert self._tlpf[stk].adns.get(loc,None) is not None, \
            ("Unindexed flavored TIDY span", (loc, sz, stk),
                self._prefs[loc], self._eva2sst[loc],
                self._tlpf[stk].adns)

  def _mark_allocated_residual(self, stk, loc, sz, isLeft):
    super()._mark_allocated_residual(stk, loc, sz, isLeft)

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

  def _alloc_place(self, stk, sz) :
    fit = self._tlpf.setdefault(stk, SegFreeList()).iterfor(sz, 1 << self._alignlog) 

    # Walk the free list to see if we have something of the "stack" flavor laying around.
    for (tpos, tsz) in fit :
        apos = self._eva_align_roundup(tpos)
        if (apos == tpos) or (tpos + tsz >= apos + sz) :
          # Remove from the freelist; any residual spans will come back to us in a moment
          self._tlpf[stk].remove(tpos)
          return apos

    # OK, that didn't work out.  Take two, go claim something without preference bigger
    # than the allocation we're tasked with, repaint it to have the current preference,
    # and return the base thereof.

    # Yes, but boy is this a slow way to get it, apparently. :(
    # Now we use per-flavor free lists.
    #
    ## for (tpos, tsz) in self._tidylst.iterfor(self._stkfaf * sz, 1 << self._alignlog) :
    ##     for (ppos, psz, pv) in self._prefs[tpos:tpos+tsz] :
    ##         if pv is not None : continue

    ##         psz = min(tpos + tsz, ppos + psz) - ppos
    ##         ppos = max(tpos, ppos)

    ##         apos = self._eva_align_roundup(ppos)
    ##         if ppos + psz >= apos + self._stkfaf * sz :
    ##           self._prefs.mark(ppos, psz, stk)
    ##           return apos

    for (tpos, tsz) in self._tlpf[None].iterfor(self._stkfaf * sz, 1 << self._alignlog) :
      apos = self._eva_align_roundup(tpos)
      if tpos + tsz >= apos + self._stkfaf * sz :
        self._prefs.mark(tpos, tsz, stk)
        self._tlpf[None].remove(tpos)
        return apos

    # OK, OK, we really haven't found anything, even if we're willing to
    # repaint.  Go grab at the wilderness; bump the wilderness pointer now to trigger
    # the common case in _mark_allocated; things will be enqueued on our per-flavor TIDY
    # list using _mark_allocated_residual
    pos = self._eva_align_roundup(self._wildern)
    self._wildern = pos + self._stkfaf * sz
    self._prefs.mark(pos, self._stkfaf * sz, stk)

    # This is kind of gross; this segment is not in any of our flavored free
    # lists, and we don't want to put it there, as our _mark_tidy would do.
    # So instead, we reach up to the superclass to update the segment state
    # and rely on duplicating any other work that our _mark_tidy does
    # (which, thankfully, is currently none)
    super()._mark_tidy(pos, self._stkfaf * sz)

    return pos

  def _mark_tidy(self, loc, sz) :
    super()._mark_tidy(loc, sz)

    for (tloc, tsz, tv) in self._prefs[loc:loc+sz] :
      nloc = max(tloc, loc)
      nsz = min(loc + sz, tloc + tsz) - nloc
      self._tlpf[tv].insert(nloc, nsz)

  def _mark_revoked(self, loc, sz) :
    # Just paint the whole thing as None, though that's potentially rude to any painted
    # spans on either end. (XXX?)

    # Remove each overlapping span from each preferrential TIDY list; the parent allocator
    # will take care of removing it from the global TIDY list.
    for (_, __, pv) in self._prefs[loc:loc+sz] :
      self._tlpf[pv].expunge(loc, sz)

    self._tlpf[None].insert(loc, sz)
    self._prefs.mark(loc, sz, None)

    super()._mark_tidy(loc,sz)

  def _maybe_revoke(self):
    # XXX configurable policy.
    #
    # Should we be looking at both the general state of the heap as well as the occupancies
    # of our preferred regions?

    if self._njunk >= self._nwait and len(self._junklru) >= 16 :
      self._do_revoke_best_and(revoke=[loc for (loc, _) in itertools.islice(self._junklru,8)])

