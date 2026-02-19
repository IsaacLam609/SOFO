open! Base
open Torch
open Maths

type nonrec dual =
  { p : any t
  ; mutable a : const t option
  }

type _ Stdlib.Effect.t +=
  | Gen1 : ((any t -> any t) * dual) -> dual Stdlib.Effect.t
  | Gen2 : ((any t -> any t -> any t) * dual * dual) -> dual Stdlib.Effect.t
  | Gen2Float : ((float -> any t -> any t) * float * dual) -> dual Stdlib.Effect.t

let const p = { p; a = None }
let primal d = d.p
let adjoint d = d.a
let lift1 f a = Stdlib.Effect.perform (Gen1 (f, a))
let lift2 f a b = Stdlib.Effect.perform (Gen2 (f, a, b))
let lift2_float f a b = Stdlib.Effect.perform (Gen2Float (f, a, b))
let ( + ) a b = lift2 ( + ) a b
let ( - ) a b = lift2 ( - ) a b
let ( * ) a b = lift2 ( * ) a b
let ( / ) a b = lift2 ( / ) a b
let ( +$ ) a b = lift2_float ( $+ ) a b
let ( -$ ) a b = lift2_float ( $- ) a b
let ( *$ ) a b = lift2_float ( $* ) a b
let ( /$ ) a b = lift2_float ( $/ ) a b
let ( *@ ) a b = lift2 ( *@ ) a b
let sigmoid a = lift1 sigmoid a
let tanh a = lift1 tanh a
let mean a = lift1 mean a
let sqr a = lift1 sqr a
let log a = lift1 log a

let eval f x =
  match f x with
  | result -> result
  | effect Gen1 (f, a), k -> Stdlib.Effect.Deep.continue k { p = f a.p; a = None }
  | effect Gen2 (f, a, b), k -> Stdlib.Effect.Deep.continue k { p = f a.p b.p; a = None }
  | effect Gen2Float (f, a, b), k ->
    Stdlib.Effect.Deep.continue k { p = f a b.p; a = None }

let __prepare a =
  let a = to_tensor a.p in
  let device = Tensor.device a in
  let a_ = Tensor.copy a |> Tensor.to_device ~device in
  let a_ = Tensor.set_requires_grad a_ ~r:true in
  Tensor.zero_grad a_;
  a_

let zero_adj p = { p; a = Some (zeros_like p) }

let update_adj x delta =
  x.a
  <- (match x.a with
      (* | None -> Some delta *)
      (* Question: do we want to distinguish between None adj (for SOFO) and zero adj (initialization)? *)
      (* Don't update adjoint if it is initialized as None *)
      | None -> None
      | Some a -> Some Maths.C.(a + delta))

let grad f x =
  match f x with
  | result, payload ->
    result.a <- Some (C.f 1.);
    result, payload
  | effect Gen1 (f, a), k ->
    let p = f a.p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      let a_ = __prepare a in
      let p = f (any (of_tensor a_)) in
      let y = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
      Tensor.backward y;
      update_adj a (of_tensor (Tensor.grad a_)));
    result
  | effect Gen2 (f, a, b), k ->
    (* prepare a for reverse pass after the continuation *)
    let p = f a.p b.p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      let a_ = __prepare a in
      let b_ = __prepare b in
      let p = f (any (of_tensor a_)) (any (of_tensor b_)) in
      let y = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
      Tensor.backward y;
      update_adj a (of_tensor (Tensor.grad a_));
      update_adj b (of_tensor (Tensor.grad b_)));
    result
  | effect Gen2Float (f, a, b), k ->
    let p = f a b.p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      let b_ = __prepare b in
      let p = f a (any (of_tensor b_)) in
      let y = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
      Tensor.backward y;
      update_adj b (of_tensor (Tensor.grad b_)));
    result

module Make (P : Prms.T) = struct
  let const p = P.map p ~f:const
  let zero_adj p = P.map p ~f:zero_adj

  (* IF we want to distinguish between None and zero adj,
     don't include the grad function here so users can handle x.a = None explicitly 
     - setting it as zeros_like could be dangerous *)
  (* let grad (f : dual P.p -> dual * 'a) (x : dual P.p) =
    let fx, payload = grad f x in
    let g =
      P.map x ~f:(fun x ->
        match x.a with
        | Some g -> g
        | None -> zeros_like x.p)
    in
    fx.p, g, payload *)
end
