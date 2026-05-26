use super::{Argument, VerifyingKey};
use crate::{
    arithmetic::CurveAffine,
    plonk::{self, ChallengeBeta, ChallengeGamma, ChallengeX, Error},
    poly::{commitment::MSM, VerifierQuery},
    transcript::{EncodedChallenge, TranscriptRead},
};
use halo2_middleware::circuit::Any;
use halo2_middleware::ff::{Field, PrimeField};
use halo2_middleware::poly::Rotation;

pub(crate) struct Committed<C: CurveAffine> {
    permutation_product_commitments: Vec<C>,
}

pub(crate) struct EvaluatedSet<C: CurveAffine> {
    permutation_product_commitment: C,
    permutation_product_eval: C::Scalar,
    permutation_product_next_eval: C::Scalar,
    permutation_product_last_eval: Option<C::Scalar>,
}

pub(crate) struct CommonEvaluated<C: CurveAffine> {
    permutation_evals: Vec<C::Scalar>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    sets: Vec<EvaluatedSet<C>>,
}

pub(crate) fn permutation_read_product_commitments<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptRead<C, E>,
>(
    arg: &Argument,
    vk: &plonk::VerifyingKey<C>,
    transcript: &mut T,
) -> Result<Committed<C>, Error> {
    let chunk_len = vk.cs_degree - 2;

    let permutation_product_commitments = arg
        .columns
        .chunks(chunk_len)
        .map(|_| transcript.read_point())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Committed {
        permutation_product_commitments,
    })
}

impl<C: CurveAffine> VerifyingKey<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        &self,
        transcript: &mut T,
    ) -> Result<CommonEvaluated<C>, Error> {
        let permutation_evals = self
            .commitments
            .iter()
            .map(|_| transcript.read_scalar())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(CommonEvaluated { permutation_evals })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(crate) fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let mut sets = vec![];

        let mut iter = self.permutation_product_commitments.into_iter();

        while let Some(permutation_product_commitment) = iter.next() {
            let permutation_product_eval = transcript.read_scalar()?;
            let permutation_product_next_eval = transcript.read_scalar()?;
            let permutation_product_last_eval = if iter.len() > 0 {
                Some(transcript.read_scalar()?)
            } else {
                None
            };

            sets.push(EvaluatedSet {
                permutation_product_commitment,
                permutation_product_eval,
                permutation_product_next_eval,
                permutation_product_last_eval,
            });
        }

        Ok(Evaluated { sets })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn expressions<'a>(
        &'a self,
        vk: &'a plonk::VerifyingKey<C>,
        p: &'a Argument,
        common: &'a CommonEvaluated<C>,
        advice_evals: &'a [C::Scalar],
        fixed_evals: &'a [C::Scalar],
        instance_evals: &'a [C::Scalar],
        l_0: C::Scalar,
        l_last: C::Scalar,
        l_blind: C::Scalar,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        x: ChallengeX<C>,
    ) -> Result<Vec<C::Scalar>, Error> {
        let chunk_len = vk.cs_degree - 2;
        let mut expressions = Vec::new();

        // Enforce only for the first set.
        // l_0(X) * (1 - z_0(X)) = 0
        if let Some(first_set) = self.sets.first() {
            expressions.push(l_0 * (C::Scalar::ONE - first_set.permutation_product_eval));
        }

        // Enforce only for the last set.
        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
        if let Some(last_set) = self.sets.last() {
            expressions.push(
                (last_set.permutation_product_eval.square() - last_set.permutation_product_eval)
                    * l_last,
            );
        }

        // Except for the first set, enforce.
        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
        for (set, last_set) in self.sets.iter().skip(1).zip(self.sets.iter()) {
            let prev_last = last_set
                .permutation_product_last_eval
                .ok_or(Error::BoundsFailure)?;
            expressions.push((set.permutation_product_eval - prev_last) * l_0);
        }

        // And for all the sets we enforce:
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z_i(\omega X) \prod (p(X) + \beta s_i(X) + \gamma)
        // - z_i(X) \prod (p(X) + \delta^i \beta X + \gamma)
        // )
        for (chunk_index, ((set, columns), permutation_evals)) in self
            .sets
            .iter()
            .zip(p.columns.chunks(chunk_len))
            .zip(common.permutation_evals.chunks(chunk_len))
            .enumerate()
        {
            let mut left = set.permutation_product_next_eval;
            for (&column, permutation_eval) in columns.iter().zip(permutation_evals.iter()) {
                let query_index = vk.cs.get_any_query_index(column, Rotation::cur())?;
                let eval = match column.column_type {
                    Any::Advice => advice_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                    Any::Fixed => fixed_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                    Any::Instance => instance_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                };
                left *= eval + (*beta * permutation_eval) + *gamma;
            }

            let mut right = set.permutation_product_eval;
            let mut current_delta = (*beta * *x)
                * (<C::Scalar as PrimeField>::DELTA
                    .pow_vartime([(chunk_index * chunk_len) as u64]));
            for &column in columns.iter() {
                let query_index = vk.cs.get_any_query_index(column, Rotation::cur())?;
                let eval = match column.column_type {
                    Any::Advice => advice_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                    Any::Fixed => fixed_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                    Any::Instance => instance_evals
                        .get(query_index)
                        .copied()
                        .ok_or(Error::BoundsFailure)?,
                };
                right *= eval + current_delta + *gamma;
                current_delta *= &C::Scalar::DELTA;
            }

            expressions.push((left - right) * (C::Scalar::ONE - (l_last + l_blind)));
        }

        Ok(expressions)
    }

    pub(in crate::plonk) fn queries<'r, M: MSM<C> + 'r>(
        &'r self,
        vk: &'r plonk::VerifyingKey<C>,
        x: ChallengeX<C>,
    ) -> Result<Vec<VerifierQuery<'r, C, M>>, Error> {
        let blinding_factors = vk.cs.blinding_factors();
        let x_next = vk.domain.rotate_omega(*x, Rotation::next());
        let x_last = vk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        let mut queries = Vec::new();
        for set in self.sets.iter() {
            // Open permutation product commitments at x and \omega^{-1} x
            // Open permutation product commitments at x and \omega x
            queries.push(VerifierQuery::new_commitment(
                &set.permutation_product_commitment,
                *x,
                set.permutation_product_eval,
            ));
            queries.push(VerifierQuery::new_commitment(
                &set.permutation_product_commitment,
                x_next,
                set.permutation_product_next_eval,
            ));
        }

        // Open it at \omega^{last} x for all but the last set
        for set in self.sets.iter().rev().skip(1) {
            queries.push(VerifierQuery::new_commitment(
                &set.permutation_product_commitment,
                x_last,
                set.permutation_product_last_eval
                    .ok_or(Error::BoundsFailure)?,
            ));
        }

        Ok(queries)
    }
}

impl<C: CurveAffine> CommonEvaluated<C> {
    pub(in crate::plonk) fn queries<'r, M: MSM<C> + 'r>(
        &'r self,
        vkey: &'r VerifyingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = VerifierQuery<'r, C, M>> + Clone {
        // Open permutation commitments for each permutation argument at x
        vkey.commitments
            .iter()
            .zip(self.permutation_evals.iter())
            .map(move |(commitment, &eval)| VerifierQuery::new_commitment(commitment, *x, eval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk::circuit::ConstraintSystemBack;
    use crate::poly::{kzg::msm::MSMKZG, EvaluationDomain};
    use group::prime::PrimeCurveAffine;
    use halo2_middleware::circuit::{Any, ColumnMid};
    use halo2curves::bn256::{Bn256, Fr, G1Affine};

    fn advice_column(index: usize) -> ColumnMid {
        ColumnMid::new(Any::Advice, index)
    }

    fn empty_cs() -> ConstraintSystemBack<Fr> {
        ConstraintSystemBack {
            num_fixed_columns: 0,
            num_advice_columns: 1,
            num_instance_columns: 0,
            num_challenges: 0,
            unblinded_advice_columns: Vec::new(),
            advice_column_phase: vec![0],
            challenge_phase: Vec::new(),
            gates: Vec::new(),
            advice_queries: Vec::new(),
            num_advice_queries: vec![0],
            instance_queries: Vec::new(),
            fixed_queries: Vec::new(),
            permutation: Argument {
                columns: vec![advice_column(0)],
            },
            lookups: Vec::new(),
            shuffles: Vec::new(),
            minimum_degree: Some(3),
        }
    }

    fn vk_with_cs(cs: ConstraintSystemBack<Fr>) -> plonk::VerifyingKey<G1Affine> {
        plonk::VerifyingKey {
            domain: EvaluationDomain::new(2, 4),
            fixed_commitments: Vec::new(),
            permutation: VerifyingKey {
                commitments: Vec::new(),
            },
            cs,
            cs_degree: 3,
            transcript_repr: Fr::zero(),
        }
    }

    fn evaluated_with_missing_previous_last_eval() -> Evaluated<G1Affine> {
        let commitment = G1Affine::identity();
        Evaluated {
            sets: vec![
                EvaluatedSet {
                    permutation_product_commitment: commitment,
                    permutation_product_eval: Fr::one(),
                    permutation_product_next_eval: Fr::one(),
                    permutation_product_last_eval: None,
                },
                EvaluatedSet {
                    permutation_product_commitment: commitment,
                    permutation_product_eval: Fr::one(),
                    permutation_product_next_eval: Fr::one(),
                    permutation_product_last_eval: None,
                },
            ],
        }
    }

    #[test]
    fn missing_permutation_last_eval_returns_error_in_expressions() {
        let vk = vk_with_cs(empty_cs());
        let evaluated = evaluated_with_missing_previous_last_eval();
        let common = CommonEvaluated {
            permutation_evals: vec![Fr::one(), Fr::one()],
        };
        let argument = Argument {
            columns: vec![advice_column(0), advice_column(0)],
        };

        let result = evaluated.expressions(
            &vk,
            &argument,
            &common,
            &[Fr::one()],
            &[],
            &[],
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            ChallengeBeta::new_for_testing(Fr::one()),
            ChallengeGamma::new_for_testing(Fr::one()),
            ChallengeX::new_for_testing(Fr::one()),
        );

        assert!(matches!(result, Err(Error::BoundsFailure)));
    }

    #[test]
    fn missing_permutation_last_eval_returns_error_in_queries() {
        let vk = vk_with_cs(empty_cs());
        let evaluated = evaluated_with_missing_previous_last_eval();

        let result =
            evaluated.queries::<MSMKZG<Bn256>>(&vk, ChallengeX::new_for_testing(Fr::one()));

        assert!(matches!(result, Err(Error::BoundsFailure)));
    }

    #[test]
    fn missing_any_query_index_returns_error() {
        let cs = empty_cs();

        let result = cs.get_any_query_index(advice_column(0), Rotation::cur());

        assert!(matches!(result, Err(Error::BoundsFailure)));
    }
}
