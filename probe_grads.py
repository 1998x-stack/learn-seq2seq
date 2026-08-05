import importlib.util
spec = importlib.util.spec_from_file_location("m01", "m01_rnn_cells/code.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import torch
device = m.device
torch.manual_seed(0)


def run_3steps(cell, is_lstm, x, h0, c0):
    """Unroll 3 steps; return scalar loss = squared L2 of final hidden(s)."""
    h = h0
    c = c0
    for t in range(3):
        if is_lstm:
            h, (h, c) = cell(x[:, t], (h, c))
        else:
            h = cell(x[:, t], h)
    if is_lstm:
        return (h ** 2).sum()
    return (h ** 2).sum()


def compare(name, mine_cls, torch_cls):
    in_s, h_s, B = 4, 16, 2
    mine = mine_cls(in_s, h_s).double().to(device)
    ref = torch_cls(in_s, h_s).double().to(device)
    # identical weights
    mine.W_ih.data.copy_(ref.weight_ih.data)
    mine.W_hh.data.copy_(ref.weight_hh.data)
    mine.b_ih.data.copy_(ref.bias_ih.data)
    mine.b_hh.data.copy_(ref.bias_hh.data)
    x = torch.randn(3, B, in_s, device=device)
    h0 = torch.randn(B, hid_s, device=device)
    c0 = torch.randn(B, hid_s, device=device)

    loss_m = run_3(mine, name == "LSTM", x, h0, c0)
    loss_r = run_3(ref, name == "LSTM", x, h0, c0)
    assert torch.allclose(loss_m, loss_r, atol=1e-8), (loss_m, loss_r)
    loss_m.backward(retain_graph=True)

    # ref must use same inputs; recompute with detached graph
    for p in ref.parameters():
        p.grad = None
    loss_r2 = run_3(ref, name == "LSTM", x, h0, c0)
    loss_r2.backward()

    wkeys = ["W_ih", "W_hh", "b_ih", "b_hh"]
    # PyTorch param names: weight_ih, weight_hh, bias_ih, bias_hh
    rmap = {"W_ih": "weight_ih", "W_hh": "weight_hh",
            "b_ih": "bias_ih", "b_hh": "bias_hh"}
    ok = True
    for wk in wkeys:
        mw = getattr(mine, wk).grad
        rw = getattr(ref, rmap[wk]).grad
        if not torch.allclose(mw, rw, atol=1e-8):
            ok = False
            print(f"  {name}.{wk} grad mismatch: max diff "
                  f"{(mw - rw).abs().max().item():.2e}")
    return ok


if __name__ == "__main__":
    print(f"device: {device}")
    for n, mc, rc in [("RNN", m.RNNCell, torch.nn.RNNCell),
                      ("LSTM", m.LSTMCell, torch.nn.LSTMCell),
                      ("GRU", m.GRUCell, torch.nn.GRUCell)]:
        ok = compare(n, mc, rc)
        print(f"{n}: gradients {'MATCH torch (PASS)' if ok else 'MISMATCH (BUG)'}")