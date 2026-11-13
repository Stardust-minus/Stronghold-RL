"""Probe variants shared by bc_probe.py-style harnesses (bc_probe2.py): SPNet2 configurations and action groups."""
VARIANTS = {
    'a': dict(width=1.0, trunk='mlp'),                    # SPNet as trained (bc1)
    'b': dict(width=2.0, trunk='mlp'),                    # 2x width
    'c': dict(width=1.0, trunk='resln', depth=4),         # deeper residual + LayerNorm trunk
    'c4': dict(width=4.0, trunk='mlp'),                   # 4x width
    'd': dict(width=1.0, trunk='mlp', attn=2, attn_dim=128, heads=4),  # entity self-attention before pooling
    'dc': dict(width=1.0, trunk='resln', depth=4, attn=2, attn_dim=128, heads=4),
    'w05': dict(width=0.5, trunk='mlp'), 'w15': dict(width=1.5, trunk='mlp'), 'w8': dict(width=8.0, trunk='mlp'),
    'e8': dict(width=1.0, trunk='mlp', emb_mult=8),       # parameters without deployed compute (folded embeddings)
    'aL': dict(width=1.0, trunk='mlp', lookahead='@lookahead'),   # + bond look-ahead
    'dL': dict(width=1.0, trunk='mlp', attn=2, attn_dim=128, heads=4, lookahead='@lookahead'),
}
GROUPS = {'scalar': (0, 4), 'buy': (4, 14), 'sell': (14, 41), 'reward': (41, 47), 'personal': (47, 52), 'card': (52, 60)}
SPNET_JS_MS = 7.761  # measured: infer.mjs, 1.32M-param SPNet, one decision (parity.py)
# version 4 (full control): cross-attention between tiles and owned pieces (d 32, one layer each way)
VARIANTS.update({
    'x': dict(width=1.0, trunk='mlp', xattn=32, qkln=True), 'bx': dict(width=2.0, trunk='mlp', xattn=32, qkln=True),
    'c4x': dict(width=4.0, trunk='mlp', xattn=32, qkln=True), 'dx': dict(width=1.0, trunk='mlp', attn=2, attn_dim=128, heads=4, xattn=32, qkln=True),
})
GROUPS4 = {**GROUPS, 'band': (60, 100), 'move': (100, 127), 'equip': (127, 154), 'tile': (154, 266), 'benchCancel': (266, 268), 'target': (268, 295)}
# version 4, in budget: self-attention over the version 3 entities only, tiles through the cross-attention (d 32)
VARIANTS.update({
    'dxs': dict(width=1.0, trunk='mlp', attn=2, attn_dim=128, heads=4, xattn=32, attn_skip=('tiles', 'globalX'), qkln=True),
    'dLxs': dict(width=1.0, trunk='mlp', attn=2, attn_dim=128, heads=4, xattn=32, attn_skip=('tiles', 'globalX'), lookahead='@lookahead', qkln=True),
    'xL': dict(width=1.0, trunk='mlp', xattn=32, lookahead='@lookahead', qkln=True),
})
