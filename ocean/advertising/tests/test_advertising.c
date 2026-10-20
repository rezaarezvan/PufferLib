#include "../advertising.h"
#include <math.h>
#include <stdio.h>

#define TEST_AGENTS 32
#define SAMPLES 400000
#define TOL 0.005f

#define CHECK(condition)                                                             \
    do {                                                                             \
        if (!(condition)) {                                                          \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
                #condition);                                                         \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CLOSE(value, want) CHECK(fabsf((value) - (want)) < TOL)

// The published model, transcribed independently of the env.
float want_cont_a[5] = {0.400000f, 0.500000f, 0.600000f, 0.700000f, 0.800000f};
float want_cont_ab[5] = {0.200000f, 0.300000f, 0.400000f, 0.500000f, 0.600000f};
float want_cont_ac[5] = {0.200000f, 0.200000f, 0.200000f, 0.200000f, 0.200000f};
float want_regress[5] = {0.400000f, 0.500000f, 0.600000f, 0.700000f, 0.800000f};
float want_half[5] = {0.300000f, 0.250000f, 0.200000f, 0.150000f, 0.100000f};
float want_third[5] = {0.200000f, 0.166667f, 0.133333f, 0.100000f, 0.066667f};
float want_cthird[5] = {0.200000f, 0.166667f, 0.133333f, 0.100000f, 0.066667f};

obs_t observations[TEST_AGENTS * OBS_SIZE];
float actions[TEST_AGENTS];
float rewards[TEST_AGENTS];
float terminals[TEST_AGENTS];
unsigned char masks[TEST_AGENTS * 7];

obs_t observations_b[TEST_AGENTS * OBS_SIZE];
float actions_b[TEST_AGENTS];
float rewards_b[TEST_AGENTS];
float terminals_b[TEST_AGENTS];
unsigned char masks_b[TEST_AGENTS * 7];

void setup(Advertising *env, int num_agents, int horizon) {
    memset(env, 0, sizeof(*env));
    env->rng = 12345;
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", num_agents);
    dict_set(&kwargs, "horizon", horizon);
    dict_set(&kwargs, "shielding", 0);
    dict_set(&kwargs, "reward_purchase", 1.0);
    dict_set(&kwargs, "reward_progress", 0.0);
    dict_set(&kwargs, "reward_lost", 0.0);
    puf_init(env, &kwargs);
    for (int a = 0; a < num_agents; a++) {
        env->agents[a].observations = observations + a * OBS_SIZE;
        env->agents[a].actions = actions + a;
        env->agents[a].rewards = rewards + a;
        env->agents[a].terminals = terminals + a;
        env->agents[a].action_mask = masks + a * 7;
    }
    puf_reset(env);
}

// Repeatedly forces advertiser 0 into a state and takes one action, tallying where
// the customer ends up.
typedef struct {
    float next[AD_STATES];
    float cost;
    float reward;
} Counts;

Counts sample(Advertising *env, int state, int action) {
    Counts counts = {0};
    Advertiser *advertiser = &env->advertisers[0];
    for (int i = 0; i < SAMPLES; i++) {
        env->tick = 0;
        env->spent = 0;
        *advertiser = (Advertiser){.state = state};
        env->agents[0].actions[0] = action;
        puf_step(env);
        counts.next[advertiser->state]++;
        counts.cost += advertiser->cost;
        counts.reward += env->agents[0].rewards[0];
    }
    for (int s = 0; s < AD_STATES; s++) {
        counts.next[s] /= SAMPLES;
    }
    counts.cost /= SAMPLES;
    counts.reward /= SAMPLES;
    return counts;
}

void test_every_row_is_a_distribution(void) {
    for (int state = 0; state < LOST; state++) {
        for (int strategy = 0; strategy < AD_STRATEGIES; strategy++) {
            int next[4];
            float probability[4];
            int outcomes = funnel_outcomes(state, strategy, next, probability);
            float total = 0.0f;
            for (int i = 0; i < outcomes; i++) {
                CHECK(probability[i] > 0.0f);
                CHECK(next[i] >= 0 && next[i] < AD_STATES);
                total += probability[i];
            }
            CHECK(fabsf(total - 1.0f) < 1e-5f);
        }
    }
}

void test_branch_a(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int state = 0; state < 5; state++) {
        for (int k = 0; k < AD_STRATEGIES; k++) {
            Counts counts = sample(&env, state, k);
            CLOSE(counts.next[state + 1], want_cont_a[k]);
            CLOSE(counts.next[LOST], want_half[k]);
            CLOSE(counts.next[state], want_half[k]);
        }
    }
}

void test_the_fork_at_state_five(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int k = 0; k < AD_STRATEGIES; k++) {
        Counts counts = sample(&env, 5, k);
        CLOSE(counts.next[6], want_cont_ab[k]);  // into branch B
        CLOSE(counts.next[10], want_cont_ac[k]); // into the branch C detour
        CLOSE(counts.next[LOST], want_half[k]);
        CLOSE(counts.next[5], want_half[k]);
    }
}

void test_branch_b_and_the_purchase_gate(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int state = 6; state < 10; state++) {
        for (int k = 0; k < AD_STRATEGIES; k++) {
            Counts counts = sample(&env, state, k);
            // State 9 advances to READY rather than to state 10, which belongs to C.
            int forward = (state == 9) ? READY : state + 1;
            CLOSE(counts.next[forward], want_cont_a[k]);
            CLOSE(counts.next[state - 1], want_third[k]);
            CLOSE(counts.next[LOST], want_third[k]);
            CLOSE(counts.next[state], want_third[k]);
        }
    }
}

void test_branch_c_entry_regresses_to_the_fork(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int k = 0; k < AD_STRATEGIES; k++) {
        Counts counts = sample(&env, 10, k);
        CLOSE(counts.next[11], want_cthird[k]);
        CLOSE(counts.next[5], want_regress[k]); // the costlier the campaign, the
        CLOSE(counts.next[LOST], want_cthird[k]); // likelier the retry at the fork
        CLOSE(counts.next[10], want_cthird[k]);
    }
}

void test_branch_c_tail(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int state = 11; state < 13; state++) {
        for (int k = 0; k < AD_STRATEGIES; k++) {
            Counts counts = sample(&env, state, k);
            CLOSE(counts.next[state - 1], want_regress[k]);
            CLOSE(counts.next[LOST], want_cthird[k]);
            CLOSE(counts.next[state], want_cthird[k]);
            CLOSE(counts.next[state + 1], want_cthird[k]);
        }
    }
}

void test_state_thirteen_loses_the_customer_twice_over(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int k = 0; k < AD_STRATEGIES; k++) {
        Counts counts = sample(&env, 13, k);
        CLOSE(counts.next[12], want_regress[k]);
        CLOSE(counts.next[LOST], 2.0f * want_cthird[k]);
        CLOSE(counts.next[13], want_cthird[k]);
    }
}

void test_a_strategy_costs_its_own_index(void) {
    Advertising env;
    setup(&env, 1, 100);
    for (int state = 0; state < LOST; state++) {
        for (int k = 0; k < AD_STRATEGIES; k++) {
            Counts counts = sample(&env, state, k);
            CLOSE(counts.cost, k);
            CLOSE(counts.reward, 0.0f);
        }
    }
}

void test_buy_pays_once_and_end_absorbs(void) {
    Advertising env;
    setup(&env, 1, 100);
    Counts counts = sample(&env, READY, BUY);
    CLOSE(counts.next[LOST], 1.0f);
    CLOSE(counts.reward, 1.0f); // the learner's unit; score reports x100
    CLOSE(counts.cost, 0.0f);
    CHECK(env.advertisers[0].bought);

    // READY is absorbing under anything but buy, and LOST under everything.
    for (int action = 0; action < 7; action++) {
        counts = sample(&env, READY, action);
        CLOSE(counts.next[action == BUY ? LOST : READY], 1.0f);
        CLOSE(counts.cost, 0.0f);
        counts = sample(&env, LOST, action);
        CLOSE(counts.next[LOST], 1.0f);
        CLOSE(counts.reward, 0.0f);
        CLOSE(counts.cost, 0.0f);
    }
}

void test_masks_follow_the_state(void) {
    Advertising env;
    setup(&env, 4, 15);
    CHECK(env.limit == 40); // L = 10 n
    for (int spent = 0; spent <= env.limit; spent += 8) {
        env.spent = spent;
        for (int state = 0; state < AD_STATES; state++) {
            env.advertisers[0].state = state;
            compute_observations(&env);
            unsigned char *mask = env.agents[0].action_mask;
            int available = 0;
            for (int action = 0; action < 7; action++) {
                CHECK(mask[action] == 0 || mask[action] == 1);
                available += mask[action];
            }
            if (state == LOST) {
                CHECK(mask[END] == 1 && available == 1);
                continue;
            }
            if (state == READY) {
                CHECK(mask[BUY] == 1 && available == 1);
                continue;
            }
            CHECK(mask[BUY] == 0 && mask[END] == 0);
            for (int k = 0; k < AD_STRATEGIES; k++) {
                CHECK(mask[k] == 1);
            }
        }
    }
}

void test_the_budget_can_be_overspent(void) {
    Advertising env;
    setup(&env, 4, 15);
    env.spent = env.limit - 2; // only two units left for four advertisers
    for (int a = 0; a < env.num_agents; a++) {
        env.agents[a].actions[0] = 4; // everyone reaches for the costliest campaign
    }
    puf_step(&env);
    CHECK(env.spent == env.limit - 2 + 4 * 4);
    CHECK(env.spent > env.limit);
    for (int a = 0; a < env.num_agents; a++) {
        CHECK(env.advertisers[a].cost == 4);
        CHECK(env.advertisers[a].strategy == 4);
    }
    for (int tick = env.tick; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = 0;
        }
        puf_step(&env);
    }
    CHECK(env.log.violation == 1.0f && env.log.n == 1.0f);
}

void test_shielded_budget_clamps_in_agent_order(void) {
    Advertising env;
    setup(&env, 4, 15);
    env.shielding = 1;
    env.spent = env.limit - 2; // only two units left for four advertisers
    for (int a = 0; a < env.num_agents; a++) {
        env.agents[a].actions[0] = 4; // everyone reaches for the costliest campaign
    }
    puf_step(&env);
    CHECK(env.spent == env.limit);
    CHECK(env.advertisers[0].cost == 2); // takes what is left, not what it asked for
    CHECK(env.advertisers[0].strategy == 2);
    for (int a = 1; a < env.num_agents; a++) {
        CHECK(env.advertisers[a].cost == 0);
        CHECK(env.advertisers[a].strategy == 0);
    }
    setup(&env, 4, 15);
    env.shielding = 1;
    for (int tick = 0; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = 4;
        }
        puf_step(&env);
        CHECK(env.spent <= env.limit);
    }
    CHECK(env.log.violation == 0.0f && env.log.n == 1.0f);
}

void test_the_budget_is_the_sum_of_costs(void) {
    Advertising env;
    setup(&env, 8, 15);
    for (int tick = 0; tick < env.horizon - 1; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % 7;
        }
        puf_step(&env);
        int total = 0;
        for (int a = 0; a < env.num_agents; a++) {
            total += env.advertisers[a].cost;
        }
        CHECK(env.spent == total);
    }
}

void test_episodes_run_exactly_the_horizon(void) {
    Advertising env;
    setup(&env, 4, 15);
    for (int episode = 0; episode < 20; episode++) {
        for (int tick = 1; tick <= env.horizon; tick++) {
            for (int a = 0; a < env.num_agents; a++) {
                env.agents[a].actions[0] = rand_r(&env.rng) % 7;
            }
            int spent = env.spent;
            puf_step(&env);
            for (int a = 0; a < env.num_agents; a++) {
                CHECK(env.agents[a].terminals[0] == (tick == env.horizon));
            }
            if (tick < env.horizon) {
                CHECK(env.tick == tick);
                CHECK(env.spent >= spent); // the budget only refills on reset
            }
        }
        CHECK(env.tick == 0); // the episode reset itself
        CHECK(env.spent == 0);
        CHECK(env.log.n == episode + 1);
        for (int a = 0; a < env.num_agents; a++) {
            CHECK(env.advertisers[a].state == 0 && env.advertisers[a].cost == 0);
        }
    }
}

void test_observations_are_well_formed(void) {
    Advertising env;
    setup(&env, 16, 15);
    for (int step = 0; step < 500; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % 7;
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            obs_t *obs = env.agents[a].observations;
            float one_hot = 0.0f;
            for (int i = 0; i < OBS_SIZE; i++) {
                float low = (i == AD_STATES + 2) ? -5.0f : 0.0f;
                CHECK(isfinite(obs[i]) && obs[i] >= low && obs[i] <= 1.0f);
            }
            for (int state = 0; state < AD_STATES; state++) {
                one_hot += obs[state];
            }
            CHECK(one_hot == 1.0f); // exactly one funnel state is marked
            CHECK(obs[AD_STATES + 3] == (float)a / (env.num_agents - 1));
        }
    }
}

void test_shaping_is_off_at_zero_and_never_moves_the_score(void) {
    Advertising env;
    setup(&env, 16, 15);
    for (int step = 0; step < 3000; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % 7;
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            // Unshaped, the only reward in the domain is the purchase.
            float reward = env.agents[a].rewards[0];
            CHECK(reward == 0.0f || reward == 1.0f);
        }
    }

    // Same seed and same actions, with shaping on. Score must be identical.
    Advertising shaped;
    setup(&shaped, 16, 15);
    for (int a = 0; a < shaped.num_agents; a++) {
        shaped.agents[a].observations = observations_b + a * OBS_SIZE;
        shaped.agents[a].actions = actions_b + a;
        shaped.agents[a].rewards = rewards_b + a;
        shaped.agents[a].terminals = terminals_b + a;
        shaped.agents[a].action_mask = masks_b + a * 7;
    }
    puf_reset(&shaped);
    shaped.reward_purchase = 1.0f;
    shaped.reward_progress = 0.1f;
    shaped.reward_lost = -0.1f;
    shaped.rng = 12345;
    env.rng = 12345;
    memset(&env.log, 0, sizeof(env.log));
    memset(&shaped.log, 0, sizeof(shaped.log));
    unsigned int actions_rng = 999;
    int saw_shaping = 0;
    for (int step = 0; step < 3000; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            float action = rand_r(&actions_rng) % 7;
            env.agents[a].actions[0] = action;
            shaped.agents[a].actions[0] = action;
        }
        puf_step(&env);
        puf_step(&shaped);
        for (int a = 0; a < env.num_agents; a++) {
            CHECK(env.advertisers[a].state == shaped.advertisers[a].state);
            CHECK(env.advertisers[a].cost == shaped.advertisers[a].cost);
            saw_shaping |= shaped.agents[a].rewards[0] != env.agents[a].rewards[0];
        }
    }
    CHECK(saw_shaping); // the shaped run really is different in reward
    CHECK(env.log.n == shaped.log.n);
    CHECK(shaped.log.score == env.log.score); // shaping never moves the objective
    CHECK(shaped.log.budget_used == env.log.budget_used);
    CHECK(shaped.log.perf == env.log.perf);
}

// Advancing one rung pays +reward_progress, slipping back pays -reward_progress,
// and being lost is priced only by its own knob.
void test_shaping_tracks_funnel_depth(void) {
    Advertising env;
    setup(&env, 1, 100);
    env.reward_progress = 0.5f;
    env.reward_lost = -3.0f;
    for (int i = 0; i < 20000; i++) {
        env.tick = 0;
        env.spent = 0;
        env.advertisers[0] = (Advertiser){.state = 7};
        env.agents[0].actions[0] = 2;
        puf_step(&env);
        int next = env.advertisers[0].state;
        float reward = env.agents[0].rewards[0];
        if (next == LOST) {
            CLOSE(reward, -3.0f);
        } else {
            CLOSE(reward, 0.5f * (DEPTH[next] - DEPTH[7]));
        }
    }
}

int main(void) {
    test_every_row_is_a_distribution();
    test_branch_a();
    test_the_fork_at_state_five();
    test_branch_b_and_the_purchase_gate();
    test_branch_c_entry_regresses_to_the_fork();
    test_branch_c_tail();
    test_state_thirteen_loses_the_customer_twice_over();
    test_a_strategy_costs_its_own_index();
    test_buy_pays_once_and_end_absorbs();
    test_masks_follow_the_state();
    test_the_budget_can_be_overspent();
    test_shielded_budget_clamps_in_agent_order();
    test_the_budget_is_the_sum_of_costs();
    test_episodes_run_exactly_the_horizon();
    test_observations_are_well_formed();
    test_shaping_is_off_at_zero_and_never_moves_the_score();
    test_shaping_tracks_funnel_depth();
    printf("All Advertising tests passed!\n");
    return 0;
}
