open! Base
open Torch
open Maths

(* For debugging *)
(* let print s = Stdio.print_endline (Sexp.to_string_hum s) *)

type nonrec dual =
  { p : t
  ; mutable a : t option
  }

type _ Stdlib.Effect.t +=
  | Gen1 : ((t -> t) * dual) -> dual Stdlib.Effect.t
  | Gen2 : ((t -> t -> t) * dual * dual) -> dual Stdlib.Effect.t
  | Gen2Float : ((t -> float -> t) * dual * float) -> dual Stdlib.Effect.t
  | Bernoulli :
      { beta : float
      ; logp : dual
      }
      -> dual Stdlib.Effect.t
  | Gumbel_Softmax :
      { tau : float
      ; hard : bool
      ; logits : dual
      }
      -> dual Stdlib.Effect.t
  | Concat :
      { dim : int
      ; x_list : dual list
      }
      -> dual Stdlib.Effect.t
  | Einsum :
      { operands : (dual * string) list
      ; return : string
      }
      -> dual Stdlib.Effect.t
  | Sum :
      { keepdim : bool option
      ; dim : int list option
      ; x : dual
      }
      -> dual Stdlib.Effect.t

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
let ( +$ ) a b = lift2_float ( +$ ) a b
let ( -$ ) a b = lift2_float ( -$ ) a b
let ( *$ ) a b = lift2_float ( *$ ) a b
let ( /$ ) a b = lift2_float ( /$ ) a b
let ( *@ ) a b = lift2 ( *@ ) a b
let neg = lift1 neg
let sigmoid = lift1 sigmoid
let tanh = lift1 tanh
let mean = lift1 mean
let sqr = lift1 sqr
let log = lift1 log
let concat ~dim x_list = Stdlib.Effect.perform (Concat { dim; x_list })
let einsum operands return = Stdlib.Effect.perform (Einsum { operands; return })
let sum ?keepdim ?dim x = Stdlib.Effect.perform (Sum { keepdim; dim; x })

module Bernoulli = struct
  let sample_primal ?(beta = 1.) (logp : dual) =
    let logp_primal = logp.p in
    let _logp = logp_primal |> Maths.primal in
    let eps = Tensor.rand_like _logp in
    let exp_logp = Tensor.(exp _logp) in
    let delta = Tensor.(exp_logp - eps) in
    (* hard thresholding *)
    let y_const = Tensor.(f 0.5 * (f 1. + sign delta)) |> Maths.const in
    (* check whether we want to propagate a tangent for forward mode *)
    let y =
      match tangent logp_primal with
      | Some _ ->
        let yt =
          Maths.(sigmoid (f beta * (exp logp_primal - Maths.const eps))) |> tangent_exn
        in
        dual ~tangent:yt y_const
      | None -> y_const
    in
    y, Maths.const exp_logp, Maths.const delta

  (* We can smooth the gradient in both logp space and p space - can experiment on it *)
  let sample_grad ~beta ~exp_logp ~delta ~(r_bar : t) =
    let open Maths in
    let tmp = exp (neg (f beta * delta)) in
    let jac = f beta * exp_logp * tmp / sqr (f 1. + tmp) in
    r_bar * jac

  let sample ?(beta = 1.) (logp : dual) : dual =
    Stdlib.Effect.perform (Bernoulli { beta; logp })
end

module Categorical = struct
  let sample_primal ~tau ~hard logits =
    let logits_primal = logits.p in
    let _logits_t = logits_primal |> Maths.primal in
    let gumbel_noise =
      let uniform_noise = Tensor.uniform _logits_t ~from:0. ~to_:1. in
      Tensor.(neg_ (log_ (neg_ (log_ uniform_noise))))
    in
    let logits_ = Tensor.(div_scalar (_logits_t + gumbel_noise) (Scalar.f tau)) in
    let shape = Tensor.shape _logits_t in
    (* Question: is this correct? *)
    (* let reduce_dim_list = List.tl_exn shape in *)
    let rank = List.length shape in
    (* Create a list [0; 1; 2; ...; rank - 1] *)
    let all_dims = List.init rank ~f:Fn.id in
    (* Filter out the 0th dimension (the batch) *)
    let reduce_dim_list = List.filter all_dims ~f:(fun d -> d <> 0) in
    let num_classes = List.nth_exn shape 1 in
    let _y =
      Tensor.(exp (logits_ - logsumexp ~dim:reduce_dim_list ~keepdim:true logits_))
    in
    let _y_final =
      if hard
      then (
        let pos = Tensor.argmax _y ~dim:1 ~keepdim:true in
        (* Question: one_hot uses Long, Only Tensors of floating point and complex dtype 
        can require gradients using set_requires_grad in Torch *)
        let one_hot = Tensor.one_hot pos ~num_classes |> Tensor.squeeze in
        Tensor.to_type one_hot ~type_:(Tensor.kind _y))
      else _y
    in
    (* check whether we want to propagate a tangent for forward mode *)
    let y =
      match tangent logits_primal with
      | None -> Maths.const _y_final
      | Some dlogits ->
        let dy =
          let tmp1 = Tensor.(div_scalar (_y * dlogits) (Scalar.f tau)) in
          let tmp2 = Tensor.(div_scalar (_y * _y * dlogits) (Scalar.f tau)) in
          Tensor.(tmp1 - tmp2)
        in
        Maths.dual ~tangent:dy (Maths.const _y_final)
    in
    y, Maths.const _y (* soft sample *)

  let sample_grad ~tau ~y_soft ~(r_bar : t) =
    let open Maths in
    let jac = (y_soft - (y_soft * y_soft)) / f tau in
    r_bar * jac

  let sample ~tau ~hard logits : dual =
    Stdlib.Effect.perform (Gumbel_Softmax { tau; hard; logits })
end

let __prepare a =
  let a = Maths.primal a.p in
  let device = Tensor.device a in
  let a_ = Tensor.copy a |> Tensor.to_device ~device in
  let a_ = Tensor.set_requires_grad a_ ~r:true in
  Tensor.zero_grad a_;
  a_

let zero_adj p = { p; a = Some (zeros_like p) }

let update_adj x delta =
  x.a
  <- (match x.a with
      | None -> None
      | Some a -> Some Maths.(a + delta))

let eval f x =
  match f x with
  | result -> result
  | effect Gen1 (f, a), k ->
    Stdlib.Effect.Deep.continue k { p = f a.p; a = None }
    (* TODO: can be replaced by const *)
  | effect Gen2 (f, a, b), k ->
    Stdlib.Effect.Deep.continue k { p = f a.p b.p; a = None }
    (* TODO: can be replaced by const *)
  | effect Gen2Float (f, a, b), k ->
    Stdlib.Effect.Deep.continue k { p = f a.p b; a = None }
    (* TODO: can be replaced by const *)
  | effect Bernoulli { beta; logp }, k ->
    let y, _, _ = Bernoulli.sample_primal ~beta logp in
    let o = const y in
    Stdlib.Effect.Deep.continue k o
  | effect Gumbel_Softmax { tau; hard; logits }, k ->
    let y, _ = Categorical.sample_primal ~tau ~hard logits in
    let o = const y in
    Stdlib.Effect.Deep.continue k o
  | effect Concat { dim; x_list }, k ->
    let x_list_p = List.map x_list ~f:(fun x -> x.p) in
    let y = Maths.concat ~dim x_list_p in
    let o = const y in
    Stdlib.Effect.Deep.continue k o
  | effect Einsum { operands; return }, k ->
    Stdlib.Effect.Deep.continue
      k
      { p = Maths.einsum (List.map operands ~f:(fun (x, eq) -> x.p, eq)) return
      ; a = None
      }
  | effect Sum { keepdim; dim; x }, k ->
    Stdlib.Effect.Deep.continue k { p = Maths.sum ?keepdim ?dim x.p; a = None }

let grad f x =
  match f x with
  | result, payload ->
    if Poly.(Maths.shape result.p <> [])
    then failwith "grad can only operate on scalar-valued functions";
    result.a <- Some (Maths.f 1.);
    result, payload
  | effect Gen1 (f, a), k ->
    let p = f a.p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      let a_ = __prepare a in
      let p = f (Maths.const a_) in
      let y = Tensor.(sum (Maths.primal r_bar * Maths.primal p)) in
      Tensor.backward y;
      update_adj a (Maths.const (Tensor.grad a_)));
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
      let p = f (Maths.const a_) (Maths.const b_) in
      let y = Tensor.(sum (Maths.primal r_bar * Maths.primal p)) in
      Tensor.backward y;
      update_adj a (Maths.const (Tensor.grad a_));
      update_adj b (Maths.const (Tensor.grad b_)));
    result
  | effect Gen2Float (f, a, b), k ->
    let p = f a.p b in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      let a_ = __prepare a in
      let p = f (Maths.const a_) b in
      let y = Tensor.(sum (Maths.primal r_bar * Maths.primal p)) in
      Tensor.backward y;
      update_adj a (Maths.const (Tensor.grad a_)));
    result
  | effect Bernoulli { beta; logp }, k ->
    let p, exp_logp, delta = Bernoulli.sample_primal ~beta logp in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    Option.iter o.a ~f:(fun r_bar ->
      let logp_bar = Bernoulli.sample_grad ~beta ~exp_logp ~delta ~r_bar in
      update_adj logp logp_bar);
    result
  | effect Gumbel_Softmax { tau; hard; logits }, k ->
    let p, p_soft = Categorical.sample_primal ~tau ~hard logits in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    Option.iter o.a ~f:(fun r_bar ->
      let logits_bar = Categorical.sample_grad ~tau ~y_soft:p_soft ~r_bar in
      update_adj logits logits_bar);
    result
  | effect Concat { dim; x_list }, k ->
    let x_list_p = List.map x_list ~f:(fun x -> x.p) in
    let p = Maths.concat ~dim x_list_p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    Option.iter o.a ~f:(fun r_bar ->
      List.fold ~init:0 x_list ~f:(fun offset x ->
        let sizes = Maths.shape x.p in
        let end_ = Int.(offset + List.nth_exn sizes dim) in
        let dx_bar = Maths.slice ~dim ~start:offset ~end_ r_bar in
        update_adj x dx_bar;
        end_)
      |> ignore);
    result
  | effect Einsum { operands; return }, k ->
    let _op = List.map operands ~f:(fun (x, eq) -> x.p, eq) in
    let p = Maths.einsum _op return in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      (* Note the code here is similar to Maths.Ops.einsum  *)
      let a = List.map ~f:fst operands in
      let a_ = List.map ~f:__prepare a in
      let equation = String.concat ~sep:"," (List.map ~f:snd operands) ^ "->" ^ return in
      let p = Tensor.einsum ~equation a_ ~path:None in
      let y = Tensor.(sum (Maths.primal r_bar * p)) in
      Tensor.backward y;
      let a__grad = List.map ~f:(fun x -> x |> Tensor.grad |> Maths.const) a_ in
      ignore (List.map2_exn a a__grad ~f:update_adj));
    result
  | effect Sum { keepdim; dim; x }, k ->
    let p = Maths.sum ?keepdim ?dim x.p in
    let o = zero_adj p in
    let result = Stdlib.Effect.Deep.continue k o in
    (* use Torch's autodiff to propagate adjoints *)
    Option.iter o.a ~f:(fun r_bar ->
      (* prepare a for reverse pass after the continuation *)
      let x_ = __prepare x in
      let p = Maths.(sum ?keepdim ?dim (const x_)) in
      let y = Tensor.(sum (Maths.primal r_bar * Maths.primal p)) in
      Tensor.backward y;
      update_adj x (Maths.const (Tensor.grad x_)));
    result

module Make (P : Prms.T) = struct
  let const p = P.map p ~f:const
  let zero_adj p = P.map p ~f:zero_adj
end
