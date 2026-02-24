open! Base
open Maths

type dual

val const : any t -> dual
val primal : dual -> any t
val adjoint : dual -> const t option
val lift1 : (any t -> any t) -> dual -> dual
val lift2 : (any t -> any t -> any t) -> dual -> dual -> dual
val lift2_float : (float -> any t -> any t) -> float -> dual -> dual
val ( + ) : dual -> dual -> dual
val ( - ) : dual -> dual -> dual
val ( * ) : dual -> dual -> dual
val ( / ) : dual -> dual -> dual
val ( +$ ) : float -> dual -> dual
val ( -$ ) : float -> dual -> dual
val ( *$ ) : float -> dual -> dual
val ( /$ ) : float -> dual -> dual
val ( *@ ) : dual -> dual -> dual
val sigmoid : dual -> dual
val tanh : dual -> dual
val mean : dual -> dual
val sqr : dual -> dual
val log : dual -> dual
val eval : ('a -> 'b) -> 'a -> 'b
val zero_adj : any t -> dual
val update_adj : dual -> const t -> unit
val grad : ('a -> dual * 'b) -> 'a -> dual * 'b

module Bernoulli : sig
  val sample : ?beta:float -> dual -> dual
end

module Make (P : Prms.T) : sig
  val const : any t P.p -> dual P.p
  val zero_adj : any t P.p -> dual P.p
  (* val grad : (dual P.p -> dual * 'a) -> dual P.p -> any t * const t P.p * 'a *)
end
