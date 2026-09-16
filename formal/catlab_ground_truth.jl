using Catlab
using Catlab.CategoricalAlgebra

@present SchLabeledGraph(FreeSchema) begin
    (V, E, L)::Ob
    src::Hom(E, V)
    tgt::Hom(E, V)
    vlabel::Hom(V, L)
end

@acset_type LabeledGraph(SchLabeledGraph, index=[:src, :tgt])

const NLABELS = 4
const A, B, C, D = 1, 2, 3, 4

function new_memory(vlabels, edges)
    g = LabeledGraph()
    add_parts!(g, :V, length(vlabels))
    add_parts!(g, :L, NLABELS)
    add_parts!(g, :E, length(edges))
    set_subpart!(g, :vlabel, vlabels)
    set_subpart!(g, :src, [s for (s, _) in edges])
    set_subpart!(g, :tgt, [t for (_, t) in edges])
    g
end

function singleton_memory(label)
    g = LabeledGraph()
    add_parts!(g, :V, 1)
    add_parts!(g, :L, NLABELS)
    set_subpart!(g, :vlabel, [label])
    g
end

mono(X, Y; V, E=Int[]) = ACSetTransformation(X, Y, V=V, E=E, L=collect(1:NLABELS))

function compose(X, Y, iface; mapX, mapY)
    po = pushout(mono(iface, X, V=mapX), mono(iface, Y, V=mapY))
    apex(po), legs(po)
end

results = Pair{String,Bool}[]
check(name, cond) = (push!(results, name => cond); nothing)

m1 = new_memory([A, B], [(1, 2)])
m2 = new_memory([B, C], [(1, 2)])
m3 = new_memory([C, D], [(1, 2)])

s12 = singleton_memory(B)
P12, (leg12_1, leg12_2) = compose(m1, m2, s12; mapX=[2], mapY=[1])
check("compose(m1,m2) identifies shared B (V=3)", nparts(P12, :V) == 3)
check("compose(m1,m2) preserves both edges (E=2)", nparts(P12, :E) == 2)
check("compose(m1,m2) keeps labels {A,B,C}", sort(collect(subpart(P12, :vlabel))) == [A, B, C])

c_in_P12 = leg12_2[:V](2)
s23 = singleton_memory(C)
P_left, _ = compose(P12, m3, s23; mapX=[c_in_P12], mapY=[1])

s23b = singleton_memory(C)
P23, (leg23_1, _) = compose(m2, m3, s23b; mapX=[2], mapY=[1])
b_in_P23 = leg23_1[:V](1)
s12b = singleton_memory(B)
P_right, _ = compose(m1, P23, s12b; mapX=[2], mapY=[b_in_P23])

check("((m1⊕m2)⊕m3) has V=4 E=3", nparts(P_left, :V) == 4 && nparts(P_left, :E) == 3)
check("(m1⊕(m2⊕m3)) has V=4 E=3", nparts(P_right, :V) == 4 && nparts(P_right, :E) == 3)
check("left labels {A,B,C,D}", sort(collect(subpart(P_left, :vlabel))) == [A, B, C, D])
check("right labels {A,B,C,D}", sort(collect(subpart(P_right, :vlabel))) == [A, B, C, D])
check("associativity: ((m1⊕m2)⊕m3) ≅ (m1⊕(m2⊕m3))", is_isomorphic(P_left, P_right))

s12c = singleton_memory(B)
P21, _ = compose(m2, m1, s12c; mapX=[1], mapY=[2])
check("commutativity: (m2⊕m1) ≅ (m1⊕m2)", is_isomorphic(P21, P12))

empty_mem = new_memory(Int[], Tuple{Int,Int}[])
s_empty = new_memory(Int[], Tuple{Int,Int}[])
po_id = pushout(mono(s_empty, m1, V=Int[]), mono(s_empty, empty_mem, V=Int[]))
check("identity: m1 ⊕ ∅ ≅ m1", is_isomorphic(apex(po_id), m1))

all_ok = all(v for (_, v) in results)
println("CATLAB GROUND TRUTH — pushout-based memory composition (Catlab v", pkgversion(Catlab), ")")
for (name, ok) in results
    println("  [", ok ? "PASS" : "FAIL", "] ", name)
end
println(all_ok ? "ALL CHECKS PASSED" : "CHECKS FAILED")
exit(all_ok ? 0 : 1)
