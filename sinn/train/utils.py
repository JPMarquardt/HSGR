def gen_to_func(y, device):
    if isinstance(y, tuple):
        return lambda x: tuple(graph_part.to(device) for graph_part in x)
    else:
        return lambda x: x.to(device)