

def reln_sexp(reln_name, poly0, poly1, color0, color1, ep):
    name = reln_name.lower()
    edge0, edge1 = ep
    vtx0, vtx1 = edge0[0], edge1[0]
    c0, c1 = color0[0].lower(), color1[0].lower()
    edge0_vtx_names = (c0 + vtx_name for vtx_name in poly0.get_edge_name_pair(edge0))
    edge1_vtx_names = (c1 + vtx_name for vtx_name in poly1.get_edge_name_pair(edge1))    
    

    edge0_name = f"EdgeFn {" ".join(edge0_vtx_names)}"
    edge1_name = f"EdgeFn {" ".join(edge1_vtx_names)}"

    if name == 'VEC':
        return f"({name} ( {c0}{poly0.get_edge_name(edge0)}, {c1}{poly1.get_vertex_name(vtx1)} ))"
    elif name == "DC":
        return "(dc ())"
    elif name == "VC":
        return f"(vc ( {vtx0}  ,{vtx1} ))"
    else:
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"


