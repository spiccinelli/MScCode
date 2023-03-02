using ITensors
using LinearAlgebra
using LBFGSB: lbfgsb
include("LBFGSBcallMPS.jl")
import Random as rnd; rnd.seed!(1234)
import Distributions as dist
using DelimitedFiles

# global dimension
NN::Int64 = 5
# local dimension
dd::Int64 = 3

# time parameters
TT::Float64 = 50.
dt::Float64 = 0.025

# Hamiltonian parameters
twopi = 2 * π
omega_1 = 5 * twopi
omega_2 = 5.5 * twopi
delta_1 = -0.35 * twopi
omega_r = 7.5 * twopi
g_1 = 0.1 * twopi
g_2 = g_1

Delta_1 = omega_1 - omega_r
Delta_2 = omega_2 - omega_r
omega_tilde_1 = omega_1 + (g_1^2 / Delta_1)
omega_tilde_2 = omega_2 + (g_2^2 / Delta_2)

JJ = (g_1 * g_2 * (Delta_1 + Delta_2)) / (Delta_1 * Delta_2)
Delta = omega_tilde_2 - omega_tilde_1

# scaling constants and MPS cutoff
e_unit::Float64 = abs(JJ)
t_unit::Float64 = 1 / e_unit
cutoff::Float64 = 1E-8

# Hamiltonian final parameters
seed = 1234
rng = rnd.MersenneTwister(seed)

perc = 0.1
Deltas = rnd.rand(dist.Normal(Delta, abs(perc * Delta)), NN) / e_unit
deltas = rnd.rand(dist.Normal(delta_1, abs(perc * delta_1)), NN) / e_unit
JJs = rnd.rand(dist.Normal(JJ, abs(perc * JJ)), NN-1) / e_unit

function inputparams(Τ::R=TT, τ::R=dt, unit::R=t_unit, info::Bool=true) where {R<:Real}    
    nt = Τ / (τ * unit)
    res = nt % 1
    nt = trunc(Integer, nt)
    nψ = nt + 1

    if info
        println("eff time window:\t", round(Τ - res * τ; digits=2), " ns")
        println("nr time steps:\t\t", nt)
    end

    return nψ, τ
end

# initialize the control vector
function initializectrls(n_iter::Int64, unit::Float64=e_unit)
    u_thr = 0.2 * 2 * π # 200 MHz
    return rnd.rand(dist.Uniform(-u_thr,u_thr), n_iter)  / unit
end

# initialize problem
nctrl, dt = inputparams()
controls = initializectrls(nctrl)
sites = siteinds("Qudit", NN; dim=dd)

# ψ states
states = ["0" for n=1:NN]
psi0 = productMPS(sites, states)

# χ states
states = ["1" for n=1:NN]
chi0 = productMPS(sites, states)

function onesitegate(jj::Int64, ctrl::Float64, σ::Vector{Index{Int64}}=sites, 
    Δs::V=Deltas, δs::V=deltas, τ::Float64=dt) where {V<:Vector{Float64}}
    one =
      ctrl * (op("a†", σ, jj) + op("a", σ, jj)) +
      Δs[jj] * op("N", σ, jj) +
      1/2 * δs[jj] * (op("N * N", σ, jj) - op("N", σ, jj))
    return exp(-im * τ/2 * one)
  end
  
  function twositegate(jj::Int64, σ::Vector{Index{Int64}}=sites, 
    κs::V=JJs, τ::Float64=dt) where {V<:Vector{Float64}}
    two =
      κs[jj] * (op("a†b", σ, jj, jj+1) + op("ab†", σ, jj, jj+1))
      return exp(-im * τ * two)
  end

  function buildgates(u_curr::Float64, u_next::Float64, nsites::Int64=NN)

    nfwd = 3 * (nsites ÷ 2)
    nbwd = 3 * ((nsites-1) ÷ 2)
    ngates = nfwd + nbwd + 2

    gates = Vector{ITensor}(undef, ngates)

    # even and odd indices
    odd = 1:2:(nsites-1)
    even = reverse(setdiff(2:nsites-1, odd))

    # forward sweep
    for (ii, jj) in enumerate(odd)
        UC_1 = onesitegate(jj, u_curr)
        UC_2 = onesitegate(jj+1, u_curr)
        UD   = twositegate(jj)

        x = (ii-1) * 3
        gates[x+1:x+3] = [UC_1, UC_2, UD]
    end

    iseven(nsites) ? u_last = u_next : u_last = u_curr
    UC_last  = onesitegate(nsites, u_last)
    gates[nfwd+1] = UC_last

    # backward sweep
    for (ii,jj) in enumerate(even)
        UD   = twositegate(jj)
        UC_3 = onesitegate(jj+1, u_next)
        UC_2 = onesitegate(jj, u_next)

        x = nfwd + 1 + (ii-1) * 3
        gates[x+1:x+3] = [UD, UC_3, UC_2]
    end

    UC_first = onesitegate(1, u_next)
    gates[ngates] = UC_first

    return gates
end

function evolvestate(ctrl::Vector{Float64}, ψini::MPS, χtgt::MPS, nsites::Int64=NN, ε::Float64=cutoff)
    
    nψ = size(ctrl,1)
    ngates = 3 * ((nsites ÷ 2) + ((nsites-1) ÷ 2)) + 2
    
    evψ = Vector{MPS}(undef, nψ)
    evχ = Vector{MPS}(undef, nψ)
    trotter = Vector{ITensor}(undef, ngates * (nψ-1))

    evψ[1] = ψini
    evχ[1] = χtgt

    for n in 1:nψ-1
        ψgates = buildgates(ctrl[n], ctrl[n+1])
        ψ = apply(ψgates, evψ[n]; cutoff=ε)
        normalize!(ψ)
        evψ[n+1] = ψ
        trotter[1+ngates*(n-1):ngates*n] = ψgates
    end

    trotter = reverse(trotter)

    for n in 1:nψ-1
        χgates = dag.(trotter[1+ngates*(n-1):ngates*n])
        χ = apply(χgates, evχ[n]; cutoff=ε)
        normalize!(χ)
        evχ[n+1] = χ
    end

    overlap = inner(first(evχ), last(evψ))

    return evψ, reverse(evχ), overlap
end

function buildMPO(σ::Vector{Index{Int64}}=sites)
    ampoMPS = OpSum()
    for jj=1:NN
        ampoMPS += "a†",jj
        ampoMPS += "a",jj
    end
    return MPO(ampoMPS,σ)  
end

ham = buildMPO()

function VNE(psi::MPS, b::Int64)
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b), siteind(psi,b+1)))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end

function entropyvonneumann(psi::MPS)
    N = length(psi)
    SvN = fill(0.0, N-1)
    for b in 1:N-1
        SvN[b] = VNE(psi, b)
    end
    return SvN
end

function jensen(ctrl::Vector{Float64}, ψini::MPS=psi0, χtgt::MPS=chi0, H::MPO=ham, τ::Float64=dt)
    N = length(ctrl)
    evψ, evχ, overlap = evolvestate(ctrl, ψini, χtgt)
    f = 1/2 * (1 - abs2(overlap))
    sandwitch = [inner(χ', H, ψ) for (χ, ψ) in zip(evχ, evψ)]
    sandwitch[1] *= 1/2
    sandwitch[N] *= 1/2
    g = real(im * conj(overlap) * sandwitch) * τ

    SvN = entropyvonneumann(last(evψ))
    fide = abs2.([inner(χtgt,psi_n) for psi_n in evψ])
    return (f, g, SvN, fide)
end

bound = 0.2 * 2 * π / e_unit
jfout, jxout, jSvN, jfide = LBFGSBcall(jensen, controls, bound, NN)
println("final fidelity: \t", 1 - last(jfout)*2)

writedlm("TNresults/jfout.csv", jfout, "\t")
writedlm("TNresults/controls.csv", controls, "\t")
writedlm("TNresults/jxout.csv", jxout, "\t")
writedlm("TNresults/jSvN.csv", jSvN, "\t")
writedlm("TNresults/jfide.csv", jfide, "\t")
