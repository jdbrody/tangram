

def reln_sexp(reln_name, poly0, poly1, color0, color1, ep):
    name = reln_name.lower()
    edge0, edge1 = ep
    vtx0, vtx1 = edge0[0], edge1[0]
    c0, c1 = color0[0].upper(), color1[0].upper()

    edge0_name = f"EdgeFn {c0}{poly0.get_edge_name(edge0)}"
    edge1_name = f"EdgeFn {c1}{poly1.get_edge_name(edge1)}"

    if name == 'VEC':
        return f"({name} ( {c0}{poly0.get_edge_name(edge0)}, {c1}{poly1.get_vertex_name(vtx1)} ))"
    elif name == "DC":
        return "(dc ())"
    elif name == "VC":
        return f"(vc ( {vtx0}  ,{vtx1} ))"
    else:
        return f"({name} ( {edge0_name}, {edge1_name} ))"


