

def reln_sexp(reln_name, poly0, poly1, color0, color1, ep):
    name = reln_name.lower()
    edge0, edge1, extra = ep
    vtx0, vtx1 = edge0[0], edge1[0]
    c0, c1 = color0[0].lower(), color1[0].lower()
    edge0_vtx_names = (c0 + vtx_name for vtx_name in poly0.get_edge_name_pair(edge0))
    edge1_vtx_names = (c1 + vtx_name for vtx_name in poly1.get_edge_name_pair(edge1))    
    

    edge0_name = f"EdgeFn {" ".join(edge0_vtx_names)}"
    edge1_name = f"EdgeFn {" ".join(edge1_vtx_names)}"

    if name == 'vec':
        return f"({name} ( {c0}{poly0.get_edge_name(edge0)}, {c1}{poly1.get_vertex_name(vtx1)} ))"
    elif name == "dc":
        return "(dc )"
    elif name == "vc":
        vtx0 = ep[2]['vmin1']
        vtx1 = ep[2]['vmin2']
        vtx0_name = f"{c0}{poly0.get_vertex_name(vtx0)}"
        vtx1_name = f"{c1}{poly1.get_vertex_name(vtx1)}"
        return f"(vc {vtx0_name}  {vtx1_name} )"
    elif name == 'teq':
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"
    elif name == 'tovi':
        #Tovi E1 E2 v1 v2 where v1 and v2 match up
        vtx0, vtx1 = extra['aligned_vertices']
        vtx0_name = f"{c0}{poly0.get_vertex_name(vtx0)}"
        vtx1_name = f"{c1}{poly1.get_vertex_name(vtx1)}"        
        return f"({name} ( {edge0_name} ) ( {edge1_name}) {vtx0_name} {vtx1_name} )"
    elif name == 'tovii':
        vtx1, vtx0 = extra['aligned_vertices']
        vtx0_name = f"{c0}{poly0.get_vertex_name(vtx0)}"
        vtx1_name = f"{c1}{poly1.get_vertex_name(vtx1)}"                
        return f"({name} ( {edge0_name} ) ( {edge1_name}) {vtx1_name} {vtx0_name} )"
    elif name == 'to':
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"
    elif name == 'tei':
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"
    elif name == 'teii':
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"
    else:
        return f"({name} ( {edge0_name} ) ( {edge1_name}) )"


