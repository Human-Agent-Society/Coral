import torch


def flashfftconv_multistream_gated_forward(
    x, k, in_gate, out_gate, fft_size
):
    batch, streams, heads, length = x.shape
    flat_x = x.reshape(batch, streams * heads, length)
    flat_k = k.reshape(streams * heads, length)
    flat_in = in_gate.reshape_as(flat_x)
    flat_out = out_gate.reshape_as(flat_x)
    x_f = torch.fft.rfft(
        flat_x.float() * flat_in.float(), n=fft_size, dim=-1
    )
    k_f = torch.fft.rfft(flat_k, n=fft_size, dim=-1)
    conv = torch.fft.irfft(
        x_f * k_f.unsqueeze(0), n=fft_size, dim=-1
    )[..., :length]
    return (
        (conv * flat_out.float())
        .to(x.dtype)
        .reshape(batch, streams, heads, length)
        .contiguous()
    )
