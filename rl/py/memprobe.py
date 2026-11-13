"""Peak GPU memory of one PPO-like minibatch step (forward + backward, KL ref forward) per arm, minibatch 8192."""
import sys, numpy as np, torch, time
from model import load
d = np.load(sys.argv[1]); I = sys.argv[2]; mb = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
dev = torch.device('cuda')
idx = np.random.default_rng(0).choice(len(d['obs']), mb)
O = torch.from_numpy(d['obs'][idx]).to(dev); M = torch.from_numpy(d['mask'][idx]).to(dev)
ref, _ = load(I + '/W1.pt', dev); ref.eval()
for arm in (sys.argv[4].split(",") if len(sys.argv) > 4 else ("W1", "W2", "AT")):
    net, _ = load(f'{I}/{arm}.pt', dev); opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.time()
    for _ in range(3):
        lg, v = net(O, M)
        with torch.no_grad(): lr_, _ = ref(O, M)
        lp = torch.log_softmax(lg, 1); loss = -(lp.clamp_min(-1e4) * torch.softmax(lr_, 1)).sum(1).mean() + v.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(arm, 'peak GB %.2f' % ((torch.cuda.max_memory_allocated() - base) / 1e9), 'step s %.3f' % ((time.time() - t0) / 3))
    del net, opt; torch.cuda.empty_cache()
