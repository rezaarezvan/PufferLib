#include "../inventory.h"
#include <math.h>
#include <stdio.h>

#define TEST_AGENTS 32
#define SAMPLES 400000
#define TOL 0.006f

#define CHECK(condition)                                                             \
    do {                                                                             \
        if (!(condition)) {                                                          \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
                #condition);                                                         \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CLOSE(value, want) CHECK(fabsf((value) - (want)) < TOL)

obs_t observations[TEST_AGENTS * OBS_SIZE];
float actions[TEST_AGENTS];
float rewards[TEST_AGENTS];
float terminals[TEST_AGENTS];
unsigned char masks[TEST_AGENTS * INV_STOCKS];

void setup(Inventory *env, int num_agents, int capacity, int horizon) {
    memset(env, 0, sizeof(*env));
    env->rng = 4242;
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", num_agents);
    dict_set(&kwargs, "capacity", capacity);
    dict_set(&kwargs, "horizon", horizon);
    dict_set(&kwargs, "shielding", 0);
    dict_set(&kwargs, "reward_scale", 1.0);
    dict_set(&kwargs, "reward_stockout", 0.0);
    puf_init(env, &kwargs);
    for (int a = 0; a < num_agents; a++) {
        env->agents[a].observations = observations + a * OBS_SIZE;
        env->agents[a].actions = actions + a;
        env->agents[a].rewards = rewards + a;
        env->agents[a].terminals = terminals + a;
        env->agents[a].action_mask = masks + a * INV_STOCKS;
    }
    puf_reset(env);
}

// Binomial(n, 1/2) written independently: C(n,k) / 2^n.
double binomial_pmf(int n, int k) {
    double coefficient = 1.0;
    for (int i = 0; i < k; i++) {
        coefficient = coefficient * (n - i) / (i + 1);
    }
    return coefficient / pow(2.0, n);
}

void test_demand_distribution(void) {
    for (int capacity = 1; capacity <= INV_MAX_CAPACITY; capacity++) {
        Inventory env;
        setup(&env, 1, capacity, 5);
        // The CDF must be the running sum of the binomial and must reach one.
        double cumulative = 0.0;
        for (int k = 0; k <= capacity; k++) {
            cumulative += binomial_pmf(capacity, k);
            CLOSE(env.demand_cdf[k], (float)cumulative);
        }
        CLOSE(env.demand_cdf[capacity], 1.0f);

        // E[min(D, k)] against a hand-rolled sum.
        for (int k = 0; k < 2 * INV_STOCKS; k++) {
            double expected = 0.0;
            for (int d = 0; d <= capacity; d++) {
                expected += binomial_pmf(capacity, d) * ((d < k) ? d : k);
            }
            CLOSE(env.expected_sales[k], (float)expected);
        }
    }
}

void test_sampled_demand_matches_the_distribution(void) {
    Inventory env;
    setup(&env, 1, 4, 1000);
    float counts[INV_STOCKS] = {0};
    Manager *manager = &env.managers[0];
    for (int i = 0; i < SAMPLES; i++) {
        env.tick = 0;
        env.spent = 0;
        *manager = (Manager){0};
        env.agents[0].actions[0] = 0;
        puf_step(&env);
        counts[manager->demand]++;
    }
    for (int d = 0; d <= env.capacity; d++) {
        CLOSE(counts[d] / SAMPLES, (float)binomial_pmf(env.capacity, d));
    }
}

void test_reward_and_cost_formulas(void) {
    Inventory env;
    setup(&env, 1, 4, 1000);
    Manager *manager = &env.managers[0];
    for (int stock = 0; stock <= env.capacity; stock++) {
        for (int order = 0; order <= env.capacity - stock; order++) {
            env.tick = 0;
            env.spent = 0;
            *manager = (Manager){.stock = stock};
            env.agents[0].actions[0] = order;
            puf_step(&env);

            // R = 4 E[min(D, s + a)] - s
            double expected = 0.0;
            for (int d = 0; d <= env.capacity; d++) {
                int available = stock + order;
                expected += binomial_pmf(env.capacity, d) * ((d < available) ? d : available);
            }
            CLOSE(env.agents[0].rewards[0], (float)(4.0 * expected - stock));
            // C = 3 1[a > 0] + 2a
            CHECK(manager->cost == (order > 0 ? 3 + 2 * order : 0));
            CHECK(env.spent == manager->cost);
        }
    }
}

void test_stock_transition(void) {
    Inventory env;
    setup(&env, 1, 6, 1000);
    Manager *manager = &env.managers[0];
    for (int i = 0; i < 20000; i++) {
        env.tick = 0;
        env.spent = 0;
        int stock = rand_r(&env.rng) % (env.capacity + 1);
        int order = rand_r(&env.rng) % (env.capacity - stock + 1);
        *manager = (Manager){.stock = stock};
        env.agents[0].actions[0] = order;
        puf_step(&env);
        int available = stock + order;
        int want = available - manager->demand;
        CHECK(manager->stock == (want > 0 ? want : 0)); // s' = max(0, s + a - D)
        CHECK(manager->stock <= env.capacity);
    }
}

void test_orders_are_capped_by_shelf_space(void) {
    Inventory env;
    setup(&env, 1, 4, 1000);
    Manager *manager = &env.managers[0];
    for (int stock = 0; stock <= env.capacity; stock++) {
        env.tick = 0;
        env.spent = 0;
        *manager = (Manager){.stock = stock};
        env.agents[0].actions[0] = env.capacity; // always ask for a full shelf
        puf_step(&env);
        CHECK(manager->order == env.capacity - stock);
    }
}

void test_the_budget_can_be_overspent(void) {
    Inventory env;
    setup(&env, 4, 4, 5);
    CHECK(env.limit == 4 * 5 * (3 + 8) / 4); // L = n H (3 + 2B) / 4
    for (int spare = 0; spare <= 12; spare++) {
        setup(&env, 4, 4, 5);
        env.spent = env.limit - spare;
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = 4; // everyone reaches for a full shelf
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            CHECK(env.managers[a].order == 4);
        }
        CHECK(env.spent == env.limit - spare + 4 * order_cost(4));
        CHECK(env.spent > env.limit);
    }
    for (int tick = env.tick; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = 0;
        }
        puf_step(&env);
    }
    CHECK(env.log.violation == 1.0f && env.log.n == 1.0f);
}

void test_shielded_budget_walks_down_to_the_largest_affordable_order(void) {
    Inventory env;
    for (int spare = 0; spare <= 12; spare++) {
        setup(&env, 4, 4, 5);
        env.shielding = 1;
        env.spent = env.limit - spare;
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = 4; // everyone reaches for a full shelf
        }
        puf_step(&env);
        // Largest order with 3 + 2a <= spare, or zero if even one unit is too dear.
        int want = 0;
        for (int order = 4; order > 0; order--) {
            if (3 + 2 * order <= spare) {
                want = order;
                break;
            }
        }
        CHECK(env.managers[0].order == want);
        CHECK(env.spent <= env.limit);
    }
    setup(&env, 4, 4, 5);
    env.shielding = 1;
    for (int tick = 0; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = env.capacity - env.managers[a].stock;
        }
        puf_step(&env);
        CHECK(env.spent <= env.limit);
    }
    CHECK(env.log.violation == 0.0f && env.log.n == 1.0f);
}

void test_the_budget_is_the_sum_of_costs(void) {
    Inventory env;
    setup(&env, 8, 4, 5);
    for (int tick = 0; tick < env.horizon - 1; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % INV_STOCKS;
        }
        puf_step(&env);
        int total = 0;
        for (int a = 0; a < env.num_agents; a++) {
            total += env.managers[a].cost;
        }
        CHECK(env.spent == total);
    }
}

void test_masks_follow_shelf_space(void) {
    Inventory env;
    setup(&env, 4, 4, 5);
    for (int spare = 0; spare <= 12; spare++) {
        for (int stock = 0; stock <= env.capacity; stock++) {
            env.spent = env.limit - spare;
            env.managers[0].stock = stock;
            compute_observations(&env);
            unsigned char *mask = env.agents[0].action_mask;
            CHECK(mask[0] == 1); // ordering nothing is always legal
            for (int order = 1; order < INV_STOCKS; order++) {
                CHECK(mask[order] == (order <= env.capacity - stock));
            }
        }
    }
}

void test_episodes_run_exactly_the_horizon(void) {
    Inventory env;
    setup(&env, 4, 4, 5);
    for (int episode = 0; episode < 20; episode++) {
        for (int tick = 1; tick <= env.horizon; tick++) {
            for (int a = 0; a < env.num_agents; a++) {
                env.agents[a].actions[0] = rand_r(&env.rng) % INV_STOCKS;
            }
            puf_step(&env);
            for (int a = 0; a < env.num_agents; a++) {
                CHECK(env.agents[a].terminals[0] == (tick == env.horizon));
            }
        }
        CHECK(env.tick == 0);
        CHECK(env.spent == 0);
        CHECK(env.log.n == episode + 1);
        for (int a = 0; a < env.num_agents; a++) {
            CHECK(env.managers[a].stock == 0 && env.managers[a].cost == 0);
        }
    }
    CHECK(env.log.perf / env.log.n >= 0.0f && env.log.perf / env.log.n <= 1.0f);
}

void test_observations_are_well_formed(void) {
    Inventory env;
    setup(&env, 16, 4, 5);
    for (int step = 0; step < 500; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % INV_STOCKS;
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            obs_t *obs = env.agents[a].observations;
            float one_hot = 0.0f;
            for (int i = 0; i < OBS_SIZE; i++) {
                float low = (i == INV_STOCKS + 2) ? -3.0f : 0.0f;
                CHECK(isfinite(obs[i]) && obs[i] >= low && obs[i] <= 1.0f);
            }
            for (int stock = 0; stock < INV_STOCKS; stock++) {
                one_hot += obs[stock];
            }
            CHECK(one_hot == 1.0f); // exactly one stock level is marked
        }
    }
}

int main(void) {
    test_demand_distribution();
    test_sampled_demand_matches_the_distribution();
    test_reward_and_cost_formulas();
    test_stock_transition();
    test_orders_are_capped_by_shelf_space();
    test_the_budget_can_be_overspent();
    test_shielded_budget_walks_down_to_the_largest_affordable_order();
    test_the_budget_is_the_sum_of_costs();
    test_masks_follow_shelf_space();
    test_episodes_run_exactly_the_horizon();
    test_observations_are_well_formed();
    printf("All Inventory tests passed!\n");
    return 0;
}
