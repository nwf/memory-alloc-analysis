#!/usr/bin/env python3

# A base class that takes care of renaming allocator objects, discarding
# map, unmap, and revocation events from the input trace.
# 
# Children should implement all their logic transitively in
#
#  _alloc(stk, tid, size) -> eva
#  _free(stk, tid, eva) -> None
#  _try_realloc(stk, eva, nsz) -> Bool
#  _maybe_revoke() -> None
#
# Children should publish their own version of events that this module
# swallows, specifically:
#
#   mapd        in _alloc, _try_realloc
#   unmapd      in _free, _try_realloc
#   revoked     in _maybe_revoke
#
# Children should not publish allocd, freed, reallocd, or size_measured
# events, leaving that to this class.  size_measured is passed through
# unmodified.

from common.misc import Publisher

class RenamingAllocatorBase (Publisher):
  __slots__ = ('_tva2eva')

  def __init__(self):
    super().__init__()
    self._tva2eva = {}

  # Alloc then publish, which is the natural order of effects
  def allocd(self, stk, tid, begin, end):
    # if begin == 0 : return  # Don't translate failing applications

    stk_delegate = stk if stk else 'malloc'
    sz = end - begin
    (eva, nsz) = self._alloc(stk_delegate, tid, sz)
    assert nsz >= sz
    self._tva2eva[begin] = eva
    self._publish('allocd', stk, tid, eva, eva+nsz)
    self._maybe_revoke()

  # Publish then free, so that effects (e.g. unmap) occur in the right order!
  def freed(self, stk, tid, begin):
    stk_delegate = stk if stk else 'free'
    eva = self._tva2eva.get(begin, None)
    if eva is not None :
      self._publish('freed', stk, tid, eva)
      self._free(stk_delegate, tid, eva)
      self._maybe_revoke()
      del self._tva2eva[begin]

  def reallocd(self, stk, tid, begin_old, begin_new, end_new):
    stk_delegate = stk if stk else 'realloc'
    szn = end_new - begin_new
    evao = self._tva2eva.get(begin_old, None)
    if evao is not None :

      if self._try_realloc(stk_delegate, tid, evao, szn) : 
        self._publish('reallocd', stk, tid, evao, evao, evao+szn)
      else :
        # otherwise, alloc new thing, free old thing
        (evan, sznn) = self._alloc(stk_delegate, tid, szn)
        self._free(stk_delegate, tid, evao)

        # Update TVA map; delete old then add new in case of equality
        del self._tva2eva[begin_old]
        self._tva2eva[begin_new] = evan

        self._publish('reallocd', stk, tid, evao, evan, evan+sznn)
    else :
      # We don't seem to have that address on file; allocate it
      # (This should include NULL, yeah?)
      self.allocd(stk, tid, begin_new, end_new)
    self._maybe_revoke()

  # Pass through
  def size_measured(self, sz):
    self._publish('size_measured', sz)

  def sweep_size_measured(self, sz):
    self._publish('sweep_size_measured', sz)

  # These are hard to pass through, so don't.  In particular, the trace
  # VAs are not linear in the emulated VA space, so a single map block
  # would translate to many spans in the emulated space.  Since, to first
  # order, we are only out to model allocator placements, and can
  # synthesize our own map and unmap events, don't pass these further
  # along the pipeline.
  def mapd(self, stk, tid, begin, end, prot):
    pass

  def unmapd(self, stk, tid, begin, end):
    pass

  def revoked(self, stk, tid, spans):
    pass
