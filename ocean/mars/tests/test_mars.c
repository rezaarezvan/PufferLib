#include "../mars.h"
#include <math.h>
#include <stdio.h>

#define TEST_AGENTS 32
#define SAMPLES 200000
#define TOL 0.01f

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CLOSE(value, want) CHECK(fabsf((value) - (want)) < TOL)

obs_t observations[TEST_AGENTS * OBS_SIZE];
float actions[TEST_AGENTS];
float rewards[TEST_AGENTS];
float terminals[TEST_AGENTS];
unsigned char masks[TEST_AGENTS * 10];

void setup(Mars *env, int num_agents, int size) {
    memset(env, 0, sizeof(*env));
    env->rng = 12345;
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", num_agents);
    dict_set(&kwargs, "size", size);
    dict_set(&kwargs, "base_seed", 0);
    dict_set(&kwargs, "num_layouts", num_agents);
    dict_set(&kwargs, "shielding", 0);
    dict_set(&kwargs, "reward_task", 1.0);
    dict_set(&kwargs, "reward_new_cell", 0.0);
    dict_set(&kwargs, "reward_lost", 0.0);
    puf_init(env, &kwargs);
    for (int a = 0; a < num_agents; a++) {
        env->agents[a].observations = observations + a * OBS_SIZE;
        env->agents[a].actions = actions + a;
        env->agents[a].rewards = rewards + a;
        env->agents[a].terminals = terminals + a;
        env->agents[a].action_mask = masks + a * 10;
    }
    puf_reset(env);
}

// Repeatedly forces rover 0 into (x, y) and takes one action, tallying where it
// ends up.
typedef struct {
    float stay;
    float done;
    float moved[4];
    float cost;
    float reward;
} Counts;

Counts sample(Mars *env, int x, int y, int action) {
    Counts counts = {0};
    Rover *rover = &env->rovers[0];
    for (int i = 0; i < SAMPLES; i++) {
        env->tick = 0;
        env->spent = 0;
        rover->x = x;
        rover->y = y;
        rover->done = 0;
        rover->cost = 0;
        env->agents[0].actions[0] = action;
        puf_step(env);
        counts.cost += rover->cost;
        counts.reward += env->agents[0].rewards[0];
        if (rover->done) {
            counts.done++;
        } else if (rover->x == x && rover->y == y) {
            counts.stay++;
        }
        for (int d = 0; d < 4; d++) {
            if (!rover->done && rover->x == x + DX[d] && rover->y == y + DY[d]) {
                counts.moved[d]++;
            }
        }
    }
    counts.stay /= SAMPLES;
    counts.done /= SAMPLES;
    counts.cost /= SAMPLES;
    counts.reward /= SAMPLES;
    for (int d = 0; d < 4; d++) {
        counts.moved[d] /= SAMPLES;
    }
    return counts;
}

void open_grid(Mars *env) {
    for (int a = 0; a < env->num_agents; a++) {
        memset(env->layouts[env->rovers[a].layout].walls, 0, MARS_CELLS);
        memset(env->layouts[env->rovers[a].layout].task_rank, 0, MARS_CELLS);
    }
}

void test_generated_layouts(void) {
    for (int size = 2; size <= MARS_MAX_GRID; size++) {
        Mars env;
        setup(&env, 1, size);
        int cells = size * size;
        Layout *rover = &env.layouts[env.rovers[0].layout];
        int walls = 0;
        int tasks = 0;
        for (int cell = 0; cell < cells; cell++) {
            walls += rover->walls[cell] > 0;
            tasks += rover->task_rank[cell] > 0;
        }
        CHECK(walls == cells * 4 / 10);
        CHECK(tasks == (cells + 9) / 10);
        CHECK(!rover->walls[0]);     // the start cell is never a wall
        CHECK(!rover->task_rank[0]); // nor a task
        CHECK(2 * number_of_reachable_cells(rover->walls, size) > cells);
        // Task rewards are 1..k, each used once, ordered by ascending Manhattan
        // distance.
        int previous = -1;
        for (int rank = 1; rank <= tasks; rank++) {
            int found = -1;
            for (int cell = 0; cell < cells; cell++) {
                if (rover->task_rank[cell] == rank) {
                    CHECK(found == -1);
                    found = cell;
                }
            }
            CHECK(found != -1);
            int distance = found % size + found / size;
            CHECK(distance >= previous);
            previous = distance;
        }
    }
}

void test_regular_move_from_an_open_cell(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    for (int action = 0; action < 4; action++) {
        Counts counts = sample(&env, 1, 1, action);
        for (int d = 0; d < 4; d++) {
            CLOSE(counts.moved[d], d == action ? 0.4f : 0.1f);
        }
        CLOSE(counts.stay, 0.1f);
        CLOSE(counts.done, 0.2f);
        CLOSE(counts.cost, 0.0f);
    }
}

void test_blocked_move_folds_onto_stay(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    Counts counts = sample(&env, 0, 0, 1);
    CLOSE(counts.stay, 0.1f + 0.4f + 0.1f);
    CLOSE(counts.moved[0], 0.1f);
    CLOSE(counts.moved[3], 0.1f);
    CLOSE(counts.done, 0.2f);

    env.layouts[env.rovers[0].layout].walls[2 * 4 + 1] = 1; // (1, 2), above (1, 1)
    counts = sample(&env, 1, 1, 0);
    CLOSE(counts.stay, 0.1f + 0.4f);
    CLOSE(counts.moved[1], 0.1f);
    CLOSE(counts.moved[2], 0.1f);
    CLOSE(counts.moved[3], 0.1f);
    CLOSE(counts.done, 0.2f);
}

void test_safe_move_costs_one_and_is_reliable(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    for (int direction = 0; direction < 4; direction++) {
        Counts counts = sample(&env, 1, 1, SAFE + direction);
        CLOSE(counts.moved[direction], 0.95f);
        CLOSE(counts.done, 0.05f);
        CLOSE(counts.cost, 1.0f); // charged whether or not the move survives
    }
}

void test_wait_holds_position(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    Counts counts = sample(&env, 1, 1, WAIT);
    CLOSE(counts.stay, 0.95f);
    CLOSE(counts.done, 0.05f);
    CLOSE(counts.cost, 0.0f);
}

void test_act_pays_the_task_rank_and_ends_the_rover(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    env.layouts[env.rovers[0].layout].task_rank[1 * 4 + 1] = 3;
    Counts counts = sample(&env, 1, 1, ACT);
    CLOSE(counts.done, 1.0f);
    CLOSE(counts.reward, 3.0f);
    CLOSE(counts.cost, 0.0f);

    // Off a task cell, act behaves as wait and pays nothing.
    counts = sample(&env, 2, 1, ACT);
    CLOSE(counts.done, 0.05f);
    CLOSE(counts.reward, 0.0f);
}

void test_done_and_walls_are_absorbing(void) {
    Mars env;
    setup(&env, 1, 4);
    open_grid(&env);
    Rover *rover = &env.rovers[0];
    Layout *layout = &env.layouts[rover->layout];
    for (int action = 0; action < 10; action++) {
        env.tick = 0;
        rover->x = 2;
        rover->y = 2;
        rover->done = 1;
        rover->cost = 0;
        env.agents[0].actions[0] = action;
        puf_step(&env);
        CHECK(rover->x == 2 && rover->y == 2 && rover->done && rover->cost == 0);
        CHECK(env.agents[0].rewards[0] == 0.0f);
    }
    layout->walls[2 * 4 + 2] = 1;
    for (int action = 0; action < 10; action++) {
        env.tick = 0;
        rover->x = 2;
        rover->y = 2;
        rover->done = 0;
        env.agents[0].actions[0] = action;
        puf_step(&env);
        CHECK(rover->x == 2 && rover->y == 2 && !rover->done);
    }
}

void test_episodes_run_exactly_the_horizon(void) {
    Mars env;
    setup(&env, 4, 4);
    CHECK(env.horizon == 8); // H = 2 size
    CHECK(env.limit == 8);   // L = H n / 4
    for (int episode = 0; episode < 20; episode++) {
        for (int tick = 1; tick <= env.horizon; tick++) {
            for (int a = 0; a < env.num_agents; a++) {
                env.agents[a].actions[0] = rand_r(&env.rng) % 10;
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
    }
}

void test_budget_is_the_sum_of_rover_costs(void) {
    Mars env;
    setup(&env, 4, 4);
    for (int tick = 0; tick < env.horizon - 1; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = SAFE + rand_r(&env.rng) % 4;
        }
        puf_step(&env);
        int total = 0;
        for (int a = 0; a < env.num_agents; a++) {
            total += env.rovers[a].cost;
        }
        CHECK(env.spent == total);
    }
}

void test_the_budget_can_be_overspent(void) {
    Mars env;
    setup(&env, 4, 4);
    open_grid(&env);
    env.spent = env.limit;
    compute_observations(&env);
    for (int a = 0; a < env.num_agents; a++) {
        for (int d = 0; d < 4; d++) {
            CHECK(env.agents[a].action_mask[SAFE + d] == 1);
            CHECK(env.agents[a].action_mask[d] == 1);
        }
        CHECK(env.agents[a].action_mask[WAIT] == 1);
    }
    for (int a = 0; a < env.num_agents; a++) {
        env.agents[a].actions[0] = SAFE;
    }
    puf_step(&env);
    CHECK(env.spent > env.limit);
    for (int a = 0; a < env.num_agents; a++) {
        CHECK(env.rovers[a].cost == 1);
    }
    for (int tick = env.tick; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = WAIT;
        }
        puf_step(&env);
    }
    CHECK(env.log.violation == 1.0f && env.log.n == 1.0f);
}

void test_shielded_safe_moves_are_masked_when_the_budget_is_dry(void) {
    Mars env;
    setup(&env, 4, 4);
    env.shielding = 1;
    open_grid(&env);
    env.spent = env.limit - 1;
    compute_observations(&env);
    for (int a = 0; a < env.num_agents; a++) {
        for (int d = 0; d < 4; d++) {
            CHECK(env.agents[a].action_mask[SAFE + d] == 1);
        }
    }
    env.spent = env.limit;
    compute_observations(&env);
    for (int a = 0; a < env.num_agents; a++) {
        for (int d = 0; d < 4; d++) {
            CHECK(env.agents[a].action_mask[SAFE + d] == 0);
            CHECK(env.agents[a].action_mask[d] == 1); // regular moves stay available
        }
        CHECK(env.agents[a].action_mask[WAIT] == 1); // wait is always available
    }
    setup(&env, 4, 4);
    env.shielding = 1;
    for (int tick = 0; tick < env.horizon; tick++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = SAFE + rand_r(&env.rng) % 4;
        }
        puf_step(&env);
        CHECK(env.spent <= env.limit);
    }
    CHECK(env.log.violation == 0.0f && env.log.n == 1.0f);
}

void test_observations_and_masks_are_well_formed(void) {
    Mars env;
    setup(&env, 8, 6);
    for (int step = 0; step < 500; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % 10;
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            obs_t *obs = env.agents[a].observations;
            float position = 0.0f;
            for (int i = 0; i < OBS_SIZE; i++) {
                float low = (i == 3 * MARS_CELLS + 3) ? -3.0f : 0.0f;
                CHECK(isfinite(obs[i]) && obs[i] >= low && obs[i] <= 1.0f);
            }
            for (int cell = 0; cell < MARS_CELLS; cell++) {
                position += obs[2 * MARS_CELLS + cell];
            }
            CHECK(position == 1.0f); // exactly one position is marked
            int available = 0;
            for (int action = 0; action < 10; action++) {
                unsigned char bit = env.agents[a].action_mask[action];
                CHECK(bit == 0 || bit == 1);
                available += bit;
            }
            CHECK(available >= 1);
        }
    }
}

void test_layout_bank_varies_and_is_reproducible(void) {
    Mars env;
    setup(&env, 4, 4);
    CHECK(env.num_layouts == 4);

    // A rover does not sit on one maze forever.
    int seen[TEST_AGENTS] = {0}; // num_layouts == num_agents in setup
    for (int episode = 0; episode < 200; episode++) {
        seen[env.rovers[0].layout] = 1;
        for (int tick = 0; tick < env.horizon; tick++) {
            for (int a = 0; a < env.num_agents; a++) {
                env.agents[a].actions[0] = WAIT;
            }
            puf_step(&env);
        }
    }
    int distinct = 0;
    for (int j = 0; j < env.num_layouts; j++) {
        distinct += seen[j];
    }
    CHECK(distinct == env.num_layouts); // every layout gets drawn

    Mars other;
    setup(&other, 4, 4);
    other.rng = 999;
    for (int j = 0; j < env.num_layouts; j++) {
        CHECK(!memcmp(env.layouts[j].walls, other.layouts[j].walls, MARS_CELLS));
        CHECK(!memcmp(env.layouts[j].task_rank, other.layouts[j].task_rank, MARS_CELLS));
    }

    // Every rover starts at the origin having visited only the origin.
    puf_reset(&env);
    for (int a = 0; a < env.num_agents; a++) {
        CHECK(env.rovers[a].x == 0 && env.rovers[a].y == 0);
        CHECK(env.rovers[a].visited[0] == 1);
        CHECK(env.rovers[a].layout >= 0 && env.rovers[a].layout < env.num_layouts);
    }
}

void test_shaping_is_off_at_zero(void) {
    Mars env;
    setup(&env, 8, 5);
    for (int step = 0; step < 2000; step++) {
        for (int a = 0; a < env.num_agents; a++) {
            env.agents[a].actions[0] = rand_r(&env.rng) % 10;
        }
        puf_step(&env);
        for (int a = 0; a < env.num_agents; a++) {
            float reward = env.agents[a].rewards[0];
            CHECK(reward == 0.0f || (reward >= 1.0f && reward <= env.max_rank));
        }
    }

    setup(&env, 1, 4);
    env.reward_new_cell = 1.0f;
    open_grid(&env);
    Rover *rover = &env.rovers[0];
    float total = 0.0f;
    for (int tick = 0; tick < 200; tick++) {
        env.tick = 0;
        rover->done = 0;
        env.agents[0].actions[0] = SAFE + (tick % 4);
        puf_step(&env);
        total += env.agents[0].rewards[0];
    }
    CHECK(total <= 16.0f); // a 4x4 grid has 16 cells and each pays once
}

int main(void) {
    test_generated_layouts();
    test_regular_move_from_an_open_cell();
    test_blocked_move_folds_onto_stay();
    test_safe_move_costs_one_and_is_reliable();
    test_wait_holds_position();
    test_act_pays_the_task_rank_and_ends_the_rover();
    test_done_and_walls_are_absorbing();
    test_episodes_run_exactly_the_horizon();
    test_budget_is_the_sum_of_rover_costs();
    test_the_budget_can_be_overspent();
    test_shielded_safe_moves_are_masked_when_the_budget_is_dry();
    test_observations_and_masks_are_well_formed();
    test_layout_bank_varies_and_is_reproducible();
    test_shaping_is_off_at_zero();
    printf("All Mars tests passed!\n");
    return 0;
}
