// ─── Deterministic replay tests for multi-actor mutations ───────────────────
//
// This module verifies that interleaved mutations across multiple actors
// (admin, creators, settler, moderator, verifier, payer, and unprivileged
// stranger) are deterministically replayable:
//
//   1. Running the same multi-actor transcript in two independent fresh `Env`
//      instances with identical initial ledger state produces identical
//      per-step outcomes, identical event topic streams, and structurally
//      identical final world snapshots.
//   2. Failing mutations (validation errors, role rejections, paused-contract
//      rejections, conflicting retries, over-refunds) roll back atomically:
//      the world snapshot after a failed step equals the snapshot before it,
//      and no events are emitted.
//   3. Retry and recovery paths (idempotent payment retries, retrying after
//      unpause, retrying after an invalid refund, and cancel/re-propose
//      ownership transfers) converge deterministically without duplicate side
//      effects.
//   4. Strict authorization (`MockAuth` without `mock_all_auths`) rejects
//      unauthorized signers—including former owners after ownership transfer—
//      and leaves state untouched.
//   5. Non-commuting orderings deterministically diverge, proving the snapshot
//      comparator captures real state differences.
//   6. Seeded multi-actor schedules preserve determinism, atomic rollback on
//      failure, and global/per-creator accounting invariants.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayActor {
    Admin,
    CreatorA,
    CreatorB,
    Settler,
    Moderator,
    Verifier,
    Payer,
    Stranger,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ReplayOp {
    AdvanceLedger(u32),
    Register {
        actor: ReplayActor,
        id: &'static str,
        price: i128,
        metadata: &'static str,
        tags: &'static [&'static str],
    },
    SetPrice {
        id: &'static str,
        new_price: i128,
    },
    SetListed {
        id: &'static str,
        listed: bool,
    },
    SetTags {
        id: &'static str,
        tags: &'static [&'static str],
    },
    ProposeTransfer {
        id: &'static str,
        new_creator: ReplayActor,
    },
    CancelTransfer {
        id: &'static str,
    },
    AcceptTransfer {
        id: &'static str,
    },
    SetVerification {
        actor: ReplayActor,
        id: &'static str,
        status: VerificationStatus,
    },
    FlagResource {
        actor: ReplayActor,
        id: &'static str,
        reason: FlagReason,
    },
    UnflagResource {
        actor: ReplayActor,
        id: &'static str,
    },
    SetPaused {
        actor: ReplayActor,
        paused: bool,
    },
    RecordPaymentIdempotent {
        settler: ReplayActor,
        receipt_id: &'static str,
        resource_id: &'static str,
        payer: ReplayActor,
        amount: i128,
        tx_hash: &'static str,
    },
    SettlePayment {
        settler: ReplayActor,
        receipt_id: &'static str,
    },
    RecordCreatorRefund {
        creator: ReplayActor,
        refund_id: &'static str,
        payment_receipt_id: &'static str,
        recipient: ReplayActor,
        amount: i128,
        tx_hash: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum StepOutcome {
    Ok,
    Err(Error),
}

#[derive(Clone, Debug, PartialEq)]
struct ResourceSnapshot {
    id: std::string::String,
    creator: ReplayActor,
    price: i128,
    metadata: std::string::String,
    listed: bool,
    state: ResourceState,
    tags: std::vec::Vec<std::string::String>,
    verified: VerificationStatus,
    frozen: bool,
    dispute_flag: DisputeFlag,
    version: u32,
    created_at: u32,
    updated_at: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct ReplayWorldSnapshot {
    ledger_sequence: u32,
    is_paused: bool,
    total_count: u32,
    listed_count: u32,
    creator_a_count: u32,
    creator_a_listed: u32,
    creator_b_count: u32,
    creator_b_listed: u32,
    creator_a_earnings: (i128, u32),
    creator_b_earnings: (i128, u32),
    resources: std::vec::Vec<ResourceSnapshot>,
    payments: std::vec::Vec<(std::string::String, PaymentReceipt)>,
    refunded_amounts: std::vec::Vec<(std::string::String, i128)>,
}

struct ReplayHarness<'a> {
    env: Env,
    client: VaultRegistryClient<'a>,
    admin: Address,
    creator_a: Address,
    creator_b: Address,
    settler: Address,
    moderator: Address,
    verifier: Address,
    payer: Address,
    stranger: Address,
}

impl<'a> ReplayHarness<'a> {
    fn new(mock_all: bool) -> Self {
        let env = Env::default();
        env.ledger().with_mut(|li| {
            li.sequence_number = 1_000;
            li.timestamp = 1_700_000_000;
        });
        if mock_all {
            env.mock_all_auths();
        }
        let contract_id = env.register(VaultRegistry, ());
        let client = VaultRegistryClient::new(&env, &contract_id);

        let admin = Address::generate(&env);
        let creator_a = Address::generate(&env);
        let creator_b = Address::generate(&env);
        let settler = Address::generate(&env);
        let moderator = Address::generate(&env);
        let verifier = Address::generate(&env);
        let payer = Address::generate(&env);
        let stranger = Address::generate(&env);

        // Bootstrap governance and operational roles deterministically.
        if mock_all {
            client.nominate_new_admin(&admin);
            client.add_settler(&settler);
            client.add_moderator(&moderator);
            client.add_verifier(&verifier);
        } else {
            client.mock_all_auths().nominate_new_admin(&admin);
            client.mock_all_auths().add_settler(&settler);
            client.mock_all_auths().add_moderator(&moderator);
            client.mock_all_auths().add_verifier(&verifier);
        }

        Self {
            env,
            client,
            admin,
            creator_a,
            creator_b,
            settler,
            moderator,
            verifier,
            payer,
            stranger,
        }
    }

    fn addr(&self, actor: ReplayActor) -> &Address {
        match actor {
            ReplayActor::Admin => &self.admin,
            ReplayActor::CreatorA => &self.creator_a,
            ReplayActor::CreatorB => &self.creator_b,
            ReplayActor::Settler => &self.settler,
            ReplayActor::Moderator => &self.moderator,
            ReplayActor::Verifier => &self.verifier,
            ReplayActor::Payer => &self.payer,
            ReplayActor::Stranger => &self.stranger,
        }
    }

    fn actor_of(&self, address: &Address) -> ReplayActor {
        if *address == self.admin {
            ReplayActor::Admin
        } else if *address == self.creator_a {
            ReplayActor::CreatorA
        } else if *address == self.creator_b {
            ReplayActor::CreatorB
        } else if *address == self.settler {
            ReplayActor::Settler
        } else if *address == self.moderator {
            ReplayActor::Moderator
        } else if *address == self.verifier {
            ReplayActor::Verifier
        } else if *address == self.payer {
            ReplayActor::Payer
        } else {
            ReplayActor::Stranger
        }
    }

    fn s(&self, value: &str) -> String {
        String::from_str(&self.env, value)
    }

    fn snapshot(&self, resource_ids: &[&str], receipt_ids: &[&str]) -> ReplayWorldSnapshot {
        let mut resources = std::vec::Vec::new();
        for id in resource_ids {
            let sid = self.s(id);
            if let Ok(Ok(r)) = self.client.try_get(&sid) {
                let mut tag_list = std::vec::Vec::new();
                for i in 0..r.tags.len() {
                    tag_list.push(r.tags.get(i).unwrap().to_string());
                }
                resources.push(ResourceSnapshot {
                    id: r.id.to_string(),
                    creator: self.actor_of(&r.creator),
                    price: r.price,
                    metadata: r.metadata.to_string(),
                    listed: r.listed,
                    state: r.state,
                    tags: tag_list,
                    verified: r.verified,
                    frozen: r.frozen,
                    dispute_flag: r.dispute_flag,
                    version: r.version,
                    created_at: r.created_at,
                    updated_at: r.updated_at,
                });
            }
        }

        let mut payments = std::vec::Vec::new();
        let mut refunded_amounts = std::vec::Vec::new();
        for rid in receipt_ids {
            let srid = self.s(rid);
            if let Ok(Ok(receipt)) = self.client.try_get_payment(&srid) {
                payments.push(((*rid).to_string(), receipt));
            }
            let refunded: i128 = self.env.as_contract(&self.client.address, || {
                self.env
                    .storage()
                    .persistent()
                    .get(&DataKey::RefundedAmount(srid.clone()))
                    .unwrap_or(0)
            });
            if refunded > 0 {
                refunded_amounts.push(((*rid).to_string(), refunded));
            }
        }

        let creator_a_earnings: (i128, u32) = self.env.as_contract(&self.client.address, || {
            self.env
                .storage()
                .persistent()
                .get(&DataKey::CreatorEarnings(self.creator_a.clone()))
                .unwrap_or((0, 0))
        });
        let creator_b_earnings: (i128, u32) = self.env.as_contract(&self.client.address, || {
            self.env
                .storage()
                .persistent()
                .get(&DataKey::CreatorEarnings(self.creator_b.clone()))
                .unwrap_or((0, 0))
        });

        ReplayWorldSnapshot {
            ledger_sequence: self.env.ledger().sequence(),
            is_paused: self.client.is_paused(),
            total_count: self.client.count(),
            listed_count: self.client.listed_count(),
            creator_a_count: self.client.creator_resource_count(&self.creator_a),
            creator_a_listed: self.client.creator_listed_count(&self.creator_a),
            creator_b_count: self.client.creator_resource_count(&self.creator_b),
            creator_b_listed: self.client.creator_listed_count(&self.creator_b),
            creator_a_earnings,
            creator_b_earnings,
            resources,
            payments,
            refunded_amounts,
        }
    }

    fn execute(&self, op: &ReplayOp) -> (StepOutcome, std::vec::Vec<std::string::String>) {
        let outcome = match op {
            ReplayOp::AdvanceLedger(delta) => {
                self.env.ledger().with_mut(|li| {
                    li.sequence_number += *delta;
                    li.timestamp += u64::from(*delta) * 5;
                });
                return (StepOutcome::Ok, std::vec::Vec::new());
            }
            ReplayOp::Register {
                actor,
                id,
                price,
                metadata,
                tags: tag_slice,
            } => map_result(self.client.try_register(
                self.addr(*actor),
                &self.s(id),
                price,
                &self.s(metadata),
                &tags(&self.env, tag_slice),
            )),
            ReplayOp::SetPrice { id, new_price } => {
                map_result(self.client.try_set_price(&self.s(id), new_price))
            }
            ReplayOp::SetListed { id, listed } => {
                map_result(self.client.try_set_listed(&self.s(id), listed))
            }
            ReplayOp::SetTags {
                id,
                tags: tag_slice,
            } => map_result(
                self.client
                    .try_set_tags(&self.s(id), &tags(&self.env, tag_slice)),
            ),
            ReplayOp::ProposeTransfer { id, new_creator } => map_result(
                self.client
                    .try_propose_transfer(&self.s(id), self.addr(*new_creator)),
            ),
            ReplayOp::CancelTransfer { id } => {
                map_result(self.client.try_cancel_transfer(&self.s(id)))
            }
            ReplayOp::AcceptTransfer { id } => {
                map_result(self.client.try_accept_transfer(&self.s(id)))
            }
            ReplayOp::SetVerification { actor, id, status } => map_result(
                self.client
                    .try_set_verification_status(&self.s(id), self.addr(*actor), status, &None),
            ),
            ReplayOp::FlagResource { actor, id, reason } => map_result(
                self.client
                    .try_flag_resource(&self.s(id), self.addr(*actor), reason),
            ),
            ReplayOp::UnflagResource { actor, id } => {
                map_result(self.client.try_unflag_resource(&self.s(id), self.addr(*actor)))
            }
            ReplayOp::SetPaused { actor, paused } => {
                map_result(self.client.try_set_paused(self.addr(*actor), paused))
            }
            ReplayOp::RecordPaymentIdempotent {
                settler,
                receipt_id,
                resource_id,
                payer,
                amount,
                tx_hash,
            } => map_result(self.client.try_record_payment_idempotent(
                self.addr(*settler),
                &self.s(receipt_id),
                &self.s(resource_id),
                self.addr(*payer),
                amount,
                &self.s(tx_hash),
            )),
            ReplayOp::SettlePayment {
                settler,
                receipt_id,
            } => map_result(
                self.client
                    .try_settle_payment(self.addr(*settler), &self.s(receipt_id)),
            ),
            ReplayOp::RecordCreatorRefund {
                creator,
                refund_id,
                payment_receipt_id,
                recipient,
                amount,
                tx_hash,
            } => map_result(self.client.try_record_creator_refund(
                self.addr(*creator),
                &self.s(refund_id),
                &self.s(payment_receipt_id),
                self.addr(*recipient),
                amount,
                &self.s(tx_hash),
            )),
        };

        let events = last_call_topics(&self.env);
        (outcome, events)
    }
}

fn map_result<T>(
    res: Result<Result<T, soroban_sdk::ConversionError>, Result<Error, soroban_sdk::InvokeError>>,
) -> StepOutcome {
    match res {
        Ok(Ok(_)) => StepOutcome::Ok,
        Err(Ok(err)) => StepOutcome::Err(err),
        Ok(Err(conv)) => panic!("unexpected conversion error in replay: {:?}", conv),
        Err(Err(inv)) => panic!("unexpected invoke error in replay: {:?}", inv),
    }
}

struct ReplayRun {
    outcomes: std::vec::Vec<StepOutcome>,
    events: std::vec::Vec<std::vec::Vec<std::string::String>>,
    final_snapshot: ReplayWorldSnapshot,
}

fn run_transcript(
    ops: &[ReplayOp],
    resource_ids: &[&str],
    receipt_ids: &[&str],
    assert_atomic_rollback: bool,
) -> ReplayRun {
    let harness = ReplayHarness::new(true);
    let mut outcomes = std::vec::Vec::with_capacity(ops.len());
    let mut events = std::vec::Vec::with_capacity(ops.len());
    let mut prev_total_count = 0u32;

    for (idx, op) in ops.iter().enumerate() {
        let before = harness.snapshot(resource_ids, receipt_ids);
        let (outcome, step_events) = harness.execute(op);
        let after = harness.snapshot(resource_ids, receipt_ids);

        if assert_atomic_rollback && matches!(outcome, StepOutcome::Err(_)) {
            assert_eq!(
                after, before,
                "step {} ({:?}) failed with {:?} but mutated state",
                idx, op, outcome
            );
            assert!(
                step_events.is_empty(),
                "step {} ({:?}) failed with {:?} but emitted events: {:?}",
                idx,
                op,
                outcome,
                step_events
            );
        }

        // Core cross-actor accounting invariants must hold after every step.
        assert_eq!(
            after.listed_count,
            after.creator_a_listed + after.creator_b_listed,
            "listed_count invariant violated at step {} ({:?})",
            idx,
            op
        );
        assert!(
            after.total_count >= prev_total_count,
            "total_count regressed at step {} ({:?})",
            idx,
            op
        );
        prev_total_count = after.total_count;

        outcomes.push(outcome);
        events.push(step_events);
    }

    let final_snapshot = harness.snapshot(resource_ids, receipt_ids);
    ReplayRun {
        outcomes,
        events,
        final_snapshot,
    }
}

#[test]
fn multi_actor_replay_normal_lifecycle_is_deterministic() {
    let ops = [
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "replayalpha",
            price: 1_000,
            metadata: "ipfs://alpha-v1",
            tags: &["ai", "prompt"],
        },
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "replaybeta",
            price: 2_500,
            metadata: "ipfs://beta-v1",
            tags: &["stellar"],
        },
        ReplayOp::AdvanceLedger(5),
        ReplayOp::SetVerification {
            actor: ReplayActor::Verifier,
            id: "replayalpha",
            status: VerificationStatus::Verified,
        },
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-alpha-1",
            resource_id: "replayalpha",
            payer: ReplayActor::Payer,
            amount: 1_000,
            tx_hash: "0xtxalpha1",
        },
        ReplayOp::SettlePayment {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-alpha-1",
        },
        ReplayOp::AdvanceLedger(3),
        ReplayOp::ProposeTransfer {
            id: "replayalpha",
            new_creator: ReplayActor::CreatorB,
        },
        ReplayOp::AcceptTransfer { id: "replayalpha" },
        ReplayOp::SetPrice {
            id: "replayalpha",
            new_price: 1_500,
        },
        ReplayOp::SetTags {
            id: "replayalpha",
            tags: &["ai", "verified"],
        },
        ReplayOp::RecordCreatorRefund {
            creator: ReplayActor::CreatorB,
            refund_id: "ref-alpha-1",
            payment_receipt_id: "rcpt-alpha-1",
            recipient: ReplayActor::Payer,
            amount: 250,
            tx_hash: "0xrefalpha1",
        },
        ReplayOp::FlagResource {
            actor: ReplayActor::Moderator,
            id: "replaybeta",
            reason: FlagReason::Spam,
        },
        ReplayOp::UnflagResource {
            actor: ReplayActor::Moderator,
            id: "replaybeta",
        },
        ReplayOp::SetListed {
            id: "replaybeta",
            listed: false,
        },
    ];

    let resource_ids = ["replayalpha", "replaybeta"];
    let receipt_ids = ["rcpt-alpha-1"];

    let run_a = run_transcript(&ops, &resource_ids, &receipt_ids, true);
    let run_b = run_transcript(&ops, &resource_ids, &receipt_ids, true);

    assert!(
        run_a.outcomes.iter().all(|o| *o == StepOutcome::Ok),
        "all normal lifecycle steps must succeed: {:?}",
        run_a.outcomes
    );
    assert_eq!(run_a.outcomes, run_b.outcomes);
    assert_eq!(run_a.events, run_b.events);
    assert_eq!(run_a.final_snapshot, run_b.final_snapshot);

    // Spot-check final ownership, listing, refund, and verification state.
    assert_eq!(run_a.final_snapshot.total_count, 2);
    assert_eq!(run_a.final_snapshot.listed_count, 1);
    assert_eq!(run_a.final_snapshot.creator_a_count, 1);
    assert_eq!(run_a.final_snapshot.creator_a_listed, 0);
    assert_eq!(run_a.final_snapshot.creator_b_count, 1);
    assert_eq!(run_a.final_snapshot.creator_b_listed, 1);
    assert_eq!(run_a.final_snapshot.resources[0].creator, ReplayActor::CreatorB);
    assert_eq!(run_a.final_snapshot.resources[0].price, 1_500);
    assert_eq!(
        run_a.final_snapshot.resources[0].verified,
        VerificationStatus::Verified
    );
    assert_eq!(
        run_a.final_snapshot.refunded_amounts,
        std::vec![("rcpt-alpha-1".to_string(), 250)]
    );
}

#[test]
fn multi_actor_replay_failure_paths_roll_back_atomically() {
    let ops = [
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "failres1",
            price: 500,
            metadata: "ipfs://fail-1",
            tags: &["core"],
        },
        // Failure 1: duplicate registration by another actor.
        ReplayOp::Register {
            actor: ReplayActor::CreatorB,
            id: "failres1",
            price: 900,
            metadata: "ipfs://dup",
            tags: &[],
        },
        // Failure 2: invalid price (zero).
        ReplayOp::SetPrice {
            id: "failres1",
            new_price: 0,
        },
        // Failure 3: unauthorized role (stranger trying to flag).
        ReplayOp::FlagResource {
            actor: ReplayActor::Stranger,
            id: "failres1",
            reason: FlagReason::Spam,
        },
        // Failure 4: unauthorized role (stranger trying to verify).
        ReplayOp::SetVerification {
            actor: ReplayActor::Stranger,
            id: "failres1",
            status: VerificationStatus::Verified,
        },
        // Failure 5: unauthorized role (stranger trying to pause).
        ReplayOp::SetPaused {
            actor: ReplayActor::Stranger,
            paused: true,
        },
        // Valid payment to set up refund & idempotency conflict checks.
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-fail-1",
            resource_id: "failres1",
            payer: ReplayActor::Payer,
            amount: 500,
            tx_hash: "0xtxfail1",
        },
        // Failure 6: conflicting idempotent payment retry (different amount).
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-fail-1",
            resource_id: "failres1",
            payer: ReplayActor::Payer,
            amount: 999,
            tx_hash: "0xtxfail1",
        },
        // Failure 7: over-refund exceeding paid amount.
        ReplayOp::RecordCreatorRefund {
            creator: ReplayActor::CreatorA,
            refund_id: "ref-over-1",
            payment_receipt_id: "rcpt-fail-1",
            recipient: ReplayActor::Payer,
            amount: 501,
            tx_hash: "0xrefover1",
        },
        // Admin pauses contract.
        ReplayOp::SetPaused {
            actor: ReplayActor::Admin,
            paused: true,
        },
        // Failure 8: payment blocked while globally paused.
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-paused-1",
            resource_id: "failres1",
            payer: ReplayActor::Payer,
            amount: 500,
            tx_hash: "0xtxpaused1",
        },
    ];

    let resource_ids = ["failres1"];
    let receipt_ids = ["rcpt-fail-1", "rcpt-paused-1"];

    let run_a = run_transcript(&ops, &resource_ids, &receipt_ids, true);
    let run_b = run_transcript(&ops, &resource_ids, &receipt_ids, true);

    assert_eq!(run_a.outcomes, run_b.outcomes);
    assert_eq!(run_a.events, run_b.events);
    assert_eq!(run_a.final_snapshot, run_b.final_snapshot);

    // Verify each expected failure returned an explicit contract error.
    assert_eq!(run_a.outcomes[1], StepOutcome::Err(Error::AlreadyRegistered));
    assert_eq!(run_a.outcomes[2], StepOutcome::Err(Error::InvalidPrice));
    assert!(matches!(run_a.outcomes[3], StepOutcome::Err(_)));
    assert!(matches!(run_a.outcomes[4], StepOutcome::Err(_)));
    assert_eq!(run_a.outcomes[5], StepOutcome::Err(Error::Unauthorized));
    assert!(matches!(run_a.outcomes[7], StepOutcome::Err(_)));
    assert!(matches!(run_a.outcomes[8], StepOutcome::Err(_)));
    assert_eq!(run_a.outcomes[10], StepOutcome::Err(Error::ContractPaused));

    // Only the single valid payment receipt exists.
    assert_eq!(run_a.final_snapshot.payments.len(), 1);
    assert!(run_a.final_snapshot.refunded_amounts.is_empty());
}

#[test]
fn multi_actor_replay_retry_and_recovery_paths_are_deterministic() {
    let ops = [
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "retryres1",
            price: 800,
            metadata: "ipfs://retry-1",
            tags: &["retry"],
        },
        // Initial payment recording.
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-retry-1",
            resource_id: "retryres1",
            payer: ReplayActor::Payer,
            amount: 800,
            tx_hash: "0xtxretry1",
        },
        ReplayOp::AdvanceLedger(10),
        // Idempotent retry with identical arguments: succeeds as a no-op with 0 events.
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-retry-1",
            resource_id: "retryres1",
            payer: ReplayActor::Payer,
            amount: 800,
            tx_hash: "0xtxretry1",
        },
        // Pause -> fail -> unpause -> retry recovery.
        ReplayOp::SetPaused {
            actor: ReplayActor::Admin,
            paused: true,
        },
        ReplayOp::SetPrice {
            id: "retryres1",
            new_price: 1_200,
        },
        ReplayOp::SetPaused {
            actor: ReplayActor::Admin,
            paused: false,
        },
        ReplayOp::SetPrice {
            id: "retryres1",
            new_price: 1_200,
        },
        // Over-refund fails, then valid partial refund retry succeeds.
        ReplayOp::RecordCreatorRefund {
            creator: ReplayActor::CreatorA,
            refund_id: "ref-retry-bad",
            payment_receipt_id: "rcpt-retry-1",
            recipient: ReplayActor::Payer,
            amount: 900,
            tx_hash: "0xrefbad",
        },
        ReplayOp::RecordCreatorRefund {
            creator: ReplayActor::CreatorA,
            refund_id: "ref-retry-ok",
            payment_receipt_id: "rcpt-retry-1",
            recipient: ReplayActor::Payer,
            amount: 300,
            tx_hash: "0xrefok",
        },
        // Transfer cancel + re-propose + accept recovery.
        ReplayOp::ProposeTransfer {
            id: "retryres1",
            new_creator: ReplayActor::Stranger,
        },
        ReplayOp::CancelTransfer { id: "retryres1" },
        ReplayOp::ProposeTransfer {
            id: "retryres1",
            new_creator: ReplayActor::CreatorB,
        },
        ReplayOp::AcceptTransfer { id: "retryres1" },
    ];

    let resource_ids = ["retryres1"];
    let receipt_ids = ["rcpt-retry-1"];

    let run_a = run_transcript(&ops, &resource_ids, &receipt_ids, true);
    let run_b = run_transcript(&ops, &resource_ids, &receipt_ids, true);

    assert_eq!(run_a.outcomes, run_b.outcomes);
    assert_eq!(run_a.events, run_b.events);
    assert_eq!(run_a.final_snapshot, run_b.final_snapshot);

    // Idempotent retry (step 3) succeeded without emitting a duplicate payment event.
    assert_eq!(run_a.outcomes[3], StepOutcome::Ok);
    assert!(run_a.events[3].is_empty());

    // Paused set_price (step 5) failed; post-unpause retry (step 7) succeeded.
    assert_eq!(run_a.outcomes[5], StepOutcome::Err(Error::ContractPaused));
    assert_eq!(run_a.outcomes[7], StepOutcome::Ok);
    assert_eq!(run_a.final_snapshot.resources[0].price, 1_200);

    // Over-refund (step 8) failed; valid retry (step 9) succeeded.
    assert!(matches!(run_a.outcomes[8], StepOutcome::Err(_)));
    assert_eq!(run_a.outcomes[9], StepOutcome::Ok);
    assert_eq!(
        run_a.final_snapshot.refunded_amounts,
        std::vec![("rcpt-retry-1".to_string(), 300)]
    );

    // Ownership converged to CreatorB after cancel + re-propose + accept.
    assert_eq!(run_a.final_snapshot.resources[0].creator, ReplayActor::CreatorB);
}

fn run_strict_auth_sequence() -> ReplayWorldSnapshot {
    let harness = ReplayHarness::new(false);
    let id = harness.s("strictauth1");
    let meta = harness.s("ipfs://strict-auth");
    let empty = empty_tags(&harness.env);

    // 1. CreatorA registers with strict MockAuth.
    let reg_invoke = build_register_invoke(
        &harness.env,
        &harness.creator_a,
        &harness.client,
        &id,
        400,
        &meta,
        &empty,
    );
    harness
        .client
        .mock_auths(&[MockAuth {
            address: &harness.creator_a,
            invoke: &reg_invoke,
        }])
        .register(&harness.creator_a, &id, &400i128, &meta, &empty);

    // 2. Stranger attempts set_price with strict MockAuth -> must panic and leave state intact.
    let before_stranger_price = harness.snapshot(&["strictauth1"], &[]);
    let stranger_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let inv = build_set_price_invoke(&harness.env, &harness.client, &id, 999);
        harness
            .client
            .mock_auths(&[MockAuth {
                address: &harness.stranger,
                invoke: &inv,
            }])
            .set_price(&id, &999i128);
    }));
    assert!(
        stranger_result.is_err(),
        "stranger must be rejected by strict auth on set_price"
    );
    assert_eq!(
        harness.snapshot(&["strictauth1"], &[]),
        before_stranger_price
    );

    // 3. CreatorA proposes transfer to CreatorB with strict MockAuth.
    let propose_inv = MockAuthInvoke {
        contract: &harness.client.address,
        fn_name: "propose_transfer",
        args: (id.clone(), harness.creator_b.clone()).into_val(&harness.env),
        sub_invokes: &[],
    };
    harness
        .client
        .mock_auths(&[MockAuth {
            address: &harness.creator_a,
            invoke: &propose_inv,
        }])
        .propose_transfer(&id, &harness.creator_b);

    // 4. Stranger attempts accept_transfer -> must panic and leave pending transfer intact.
    let before_stranger_accept = harness.snapshot(&["strictauth1"], &[]);
    let stranger_accept = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let inv = MockAuthInvoke {
            contract: &harness.client.address,
            fn_name: "accept_transfer",
            args: (id.clone(),).into_val(&harness.env),
            sub_invokes: &[],
        };
        harness
            .client
            .mock_auths(&[MockAuth {
                address: &harness.stranger,
                invoke: &inv,
            }])
            .accept_transfer(&id);
    }));
    assert!(
        stranger_accept.is_err(),
        "stranger must be rejected by strict auth on accept_transfer"
    );
    assert_eq!(
        harness.snapshot(&["strictauth1"], &[]),
        before_stranger_accept
    );

    // 5. CreatorB (pending owner) accepts transfer with strict MockAuth.
    let accept_inv = MockAuthInvoke {
        contract: &harness.client.address,
        fn_name: "accept_transfer",
        args: (id.clone(),).into_val(&harness.env),
        sub_invokes: &[],
    };
    harness
        .client
        .mock_auths(&[MockAuth {
            address: &harness.creator_b,
            invoke: &accept_inv,
        }])
        .accept_transfer(&id);

    // 6. Post-transfer revocation: former owner CreatorA can no longer mutate price.
    let before_former_owner = harness.snapshot(&["strictauth1"], &[]);
    let former_owner_attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let inv = build_set_price_invoke(&harness.env, &harness.client, &id, 777);
        harness
            .client
            .mock_auths(&[MockAuth {
                address: &harness.creator_a,
                invoke: &inv,
            }])
            .set_price(&id, &777i128);
    }));
    assert!(
        former_owner_attempt.is_err(),
        "former owner must be rejected after ownership transfer"
    );
    assert_eq!(harness.snapshot(&["strictauth1"], &[]), before_former_owner);

    // 7. New owner CreatorB succeeds on set_price with strict MockAuth.
    let new_owner_inv = build_set_price_invoke(&harness.env, &harness.client, &id, 650);
    harness
        .client
        .mock_auths(&[MockAuth {
            address: &harness.creator_b,
            invoke: &new_owner_inv,
        }])
        .set_price(&id, &650i128);

    harness.snapshot(&["strictauth1"], &[])
}

#[test]
fn multi_actor_replay_strict_auth_enforcement_and_post_transfer_revocation() {
    let snap_a = run_strict_auth_sequence();
    let snap_b = run_strict_auth_sequence();

    assert_eq!(snap_a, snap_b);
    assert_eq!(snap_a.resources.len(), 1);
    assert_eq!(snap_a.resources[0].creator, ReplayActor::CreatorB);
    assert_eq!(snap_a.resources[0].price, 650);
    assert_eq!(snap_a.creator_a_count, 0);
    assert_eq!(snap_a.creator_b_count, 1);
}

#[test]
fn multi_actor_replay_detects_non_commuting_order_divergence() {
    let order_1 = [
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "orderres1",
            price: 100,
            metadata: "ipfs://order",
            tags: &[],
        },
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-order-1",
            resource_id: "orderres1",
            payer: ReplayActor::Payer,
            amount: 100,
            tx_hash: "0xtxorder1",
        },
        ReplayOp::SetPaused {
            actor: ReplayActor::Admin,
            paused: true,
        },
    ];

    let order_2 = [
        ReplayOp::Register {
            actor: ReplayActor::CreatorA,
            id: "orderres1",
            price: 100,
            metadata: "ipfs://order",
            tags: &[],
        },
        ReplayOp::SetPaused {
            actor: ReplayActor::Admin,
            paused: true,
        },
        ReplayOp::RecordPaymentIdempotent {
            settler: ReplayActor::Settler,
            receipt_id: "rcpt-order-1",
            resource_id: "orderres1",
            payer: ReplayActor::Payer,
            amount: 100,
            tx_hash: "0xtxorder1",
        },
    ];

    let r1_a = run_transcript(&order_1, &["orderres1"], &["rcpt-order-1"], true);
    let r1_b = run_transcript(&order_1, &["orderres1"], &["rcpt-order-1"], true);
    assert_eq!(r1_a.final_snapshot, r1_b.final_snapshot);

    let r2_a = run_transcript(&order_2, &["orderres1"], &["rcpt-order-1"], true);
    let r2_b = run_transcript(&order_2, &["orderres1"], &["rcpt-order-1"], true);
    assert_eq!(r2_a.final_snapshot, r2_b.final_snapshot);

    // Non-commuting orderings must diverge deterministically.
    assert_ne!(r1_a.final_snapshot, r2_a.final_snapshot);
    assert_eq!(r1_a.final_snapshot.payments.len(), 1);
    assert_eq!(r2_a.final_snapshot.payments.len(), 0);
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn build_seeded_schedule(seed: u64) -> std::vec::Vec<ReplayOp> {
    let mut rng = seed;
    let ids = ["seedres0", "seedres1", "seedres2"];
    let mut ops = std::vec::Vec::new();

    // Seed initial resources across both creators.
    ops.push(ReplayOp::Register {
        actor: ReplayActor::CreatorA,
        id: ids[0],
        price: 500,
        metadata: "ipfs://seed-0",
        tags: &["seed"],
    });
    ops.push(ReplayOp::Register {
        actor: ReplayActor::CreatorB,
        id: ids[1],
        price: 750,
        metadata: "ipfs://seed-1",
        tags: &["seed"],
    });

    for step in 0..24u32 {
        let r = next_u64(&mut rng);
        let id = ids[(r as usize) % ids.len()];
        match (r >> 8) % 9 {
            0 => ops.push(ReplayOp::AdvanceLedger(1 + ((r >> 16) % 5) as u32)),
            1 => ops.push(ReplayOp::Register {
                actor: if (r & 1) == 0 {
                    ReplayActor::CreatorA
                } else {
                    ReplayActor::CreatorB
                },
                id,
                price: 300 + ((r >> 16) % 500) as i128,
                metadata: "ipfs://seed-new",
                tags: &["gen"],
            }),
            2 => {
                // Mix valid prices and intentionally invalid zero price.
                let price = if (r >> 24) % 4 == 0 {
                    0
                } else {
                    100 + ((r >> 16) % 2_000) as i128
                };
                ops.push(ReplayOp::SetPrice {
                    id,
                    new_price: price,
                });
            }
            3 => ops.push(ReplayOp::SetListed {
                id,
                listed: ((r >> 16) & 1) == 0,
            }),
            4 => ops.push(ReplayOp::SetVerification {
                actor: if (r >> 24) % 4 == 0 {
                    ReplayActor::Stranger
                } else {
                    ReplayActor::Verifier
                },
                id,
                status: VerificationStatus::Verified,
            }),
            5 => ops.push(ReplayOp::FlagResource {
                actor: if (r >> 24) % 4 == 0 {
                    ReplayActor::Stranger
                } else {
                    ReplayActor::Moderator
                },
                id,
                reason: FlagReason::Spam,
            }),
            6 => ops.push(ReplayOp::UnflagResource {
                actor: ReplayActor::Moderator,
                id,
            }),
            7 => ops.push(ReplayOp::SetPaused {
                actor: if (r >> 24) % 4 == 0 {
                    ReplayActor::Stranger
                } else {
                    ReplayActor::Admin
                },
                paused: (step % 2) == 0,
            }),
            _ => ops.push(ReplayOp::RecordPaymentIdempotent {
                settler: ReplayActor::Settler,
                receipt_id: "rcpt-seed-0",
                resource_id: ids[0],
                payer: ReplayActor::Payer,
                amount: 500,
                tx_hash: "0xtxseed0",
            }),
        }
    }

    ops
}

#[test]
fn multi_actor_replay_seeded_schedules_preserve_determinism_and_invariants() {
    let seeds = [0x1234_5678_9abc_def0u64, 0xdead_beef_cafe_babe, 0x0bad_f00d_1337_4242, 42, 999_999];
    let resource_ids = ["seedres0", "seedres1", "seedres2"];
    let receipt_ids = ["rcpt-seed-0"];

    for seed in seeds {
        let ops = build_seeded_schedule(seed);
        let run_a = run_transcript(&ops, &resource_ids, &receipt_ids, true);
        let run_b = run_transcript(&ops, &resource_ids, &receipt_ids, true);

        assert_eq!(
            run_a.outcomes, run_b.outcomes,
            "outcomes diverged for seed {:#x}",
            seed
        );
        assert_eq!(
            run_a.events, run_b.events,
            "event stream diverged for seed {:#x}",
            seed
        );
        assert_eq!(
            run_a.final_snapshot, run_b.final_snapshot,
            "final snapshot diverged for seed {:#x}",
            seed
        );
    }
}
