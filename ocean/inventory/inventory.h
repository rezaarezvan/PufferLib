#include "raylib.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"

#ifndef INV_MAX_CAPACITY
#define INV_MAX_CAPACITY 10
#endif
#if INV_MAX_CAPACITY < 1 || INV_MAX_CAPACITY > 64
#error "INV_MAX_CAPACITY must be in [1, 64]"
#endif
#define INV_STOCKS (INV_MAX_CAPACITY + 1)

#define ACT_SIZES {INV_STOCKS}
#define OBS_SIZE (INV_STOCKS + 4)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 6

// Marthe et al.'s stage profit, with procurement split out so operating reward and shared spend are two separate signals.
// R = 4 E[min(D, s+a)] - s
// C = 3 1[a > 0] + 2a.
// The fixed 3 is what makes cost non-monotone in the action index.
const float SALES_VALUE = 4.0f;
const int ORDER_FIXED_COST = 3;
const int ORDER_UNIT_COST = 2;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_GOLD = (Color){235, 185, 70, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_EDGE = (Color){20, 52, 52, 255};

// Required struct. Only use floats!
struct Log {
    float perf;  // realised sales as a fraction of demand
    float score; // team operating reward, the domain objective, unshaped
    float episode_return;
    float episode_length;
    float budget_used;  // team spend as a fraction of the limit, over 1 if overspent
    float spend_spread; // widest gap between two managers, over the fair share
    float violation;    // Pr(C > L)
    float n;            // Required as the last field
};

typedef struct Manager Manager;
struct Manager {
    int stock;
    int cost;
    int order;  // what it actually ordered last step, for the render
    int demand; // what it actually faced last step, for the render
    float ret;
};

// Required that you have some struct for your env
struct Env {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Agent *agents;
    int tag;
    int boundary_reached;
    int num_agents;
    Manager *managers;
    int shielding;
    int capacity;
    int horizon;
    int tick;
    int spent;
    int limit;
    float reward_scale;
    float reward_stockout;
    // Demand is Binomial(capacity, 1/2), fixed for the whole run, so the CDF and
    // the expected-sales table are built once at init and only read after.
    float demand_cdf[INV_STOCKS];
    float expected_sales[2 * INV_STOCKS];
    float total_demand;
    float total_sold;
    Texture2D puffer;
    unsigned int rng;
};
typedef Env Inventory;

int order_cost(int order) {
    return (order > 0) ? ORDER_FIXED_COST + ORDER_UNIT_COST * order : 0;
}

void compute_observations(Inventory *env) {
    int remaining = env->limit - env->spent;
    for (int a = 0; a < env->num_agents; a++) {
        Manager *manager = &env->managers[a];
        obs_t *obs = env->agents[a].observations;
        memset(obs, 0, OBS_SIZE * sizeof(obs_t));
        obs[manager->stock] = 1.0f;
        obs[INV_STOCKS + 0] = (float)env->tick / (float)env->horizon;
        obs[INV_STOCKS + 1] = (float)manager->cost / (float)env->limit;
        obs[INV_STOCKS + 2] = (float)remaining / (float)env->limit;
        obs[INV_STOCKS + 3] =
            (env->num_agents > 1) ? (float)a / (env->num_agents - 1) : 0.0f;

        // Shelf space caps the order, and shielding caps it again by cost.
        // Ordering nothing is free, so it stays legal either way.
        unsigned char *mask = env->agents[a].action_mask;
        memset(mask, 0, INV_STOCKS);
        for (int order = 0; order <= env->capacity - manager->stock; order++) {
            mask[order] = !env->shielding || order_cost(order) <= remaining;
        }
    }
}

// Required function
void puf_reset(Inventory *env) {
    env->tick = 0;
    env->spent = 0;
    env->total_demand = 0.0f;
    env->total_sold = 0.0f;
    for (int a = 0; a < env->num_agents; a++) {
        env->managers[a] = (Manager){0};
    }
    compute_observations(env);
}

void add_log(Inventory *env) {
    float agents = env->num_agents;
    float ret = 0.0f;
    int most = 0;
    int least = env->limit;
    for (int a = 0; a < env->num_agents; a++) {
        Manager *manager = &env->managers[a];
        ret += manager->ret;
        most = (manager->cost > most) ? manager->cost : most;
        least = (manager->cost < least) ? manager->cost : least;
    }
    // Fill rate: units actually sold over units demanded.
    env->log.perf +=
        (env->total_demand > 0.0f) ? env->total_sold / env->total_demand : 0.0f;
    env->log.score += ret;
    env->log.episode_return += ret / agents;
    env->log.episode_length += env->horizon;
    env->log.budget_used += (float)env->spent / (float)env->limit;
    env->log.spend_spread += (float)(most - least) * agents / (float)env->limit;
    env->log.violation += env->spent > env->limit;
    env->log.n++;
}

// Required function
void puf_step(Inventory *env) {
    for (int a = 0; a < env->num_agents; a++) {
        env->agents[a].rewards[0] = 0.0f;
        env->agents[a].terminals[0] = 0.0f;
    }
    env->tick++;

    for (int a = 0; a < env->num_agents; a++) {
        Manager *manager = &env->managers[a];
        int order = env->agents[a].actions[0];
        if (order > env->capacity - manager->stock) {
            order = env->capacity - manager->stock;
        }
        // Cost is not monotone in the action index, so a short budget walks
        // down to the largest affordable order rather than clamping the index.
        int remaining = env->limit - env->spent;
        while (env->shielding && order > 0 && order_cost(order) > remaining) {
            order--;
        }
        manager->cost += order_cost(order);
        env->spent += order_cost(order);
        manager->order = order;

        int available = manager->stock + order;
        float reward = SALES_VALUE * env->expected_sales[available] - manager->stock;
        manager->ret += reward;
        env->agents[a].rewards[0] = env->reward_scale * reward;

        float draw = (float)rand_r(&env->rng) / (float)RAND_MAX;
        int demand = 0;
        while (demand < env->capacity && draw >= env->demand_cdf[demand]) {
            demand++;
        }
        manager->demand = demand;
        env->total_demand += demand;
        env->total_sold += (demand < available) ? demand : available;
        if (demand >= available) {
            env->agents[a].rewards[0] += env->reward_stockout;
        }
        manager->stock = (available > demand) ? available - demand : 0;
    }

    if (env->tick < env->horizon) {
        compute_observations(env);
        return;
    }
    add_log(env);
    puf_reset(env);
    for (int a = 0; a < env->num_agents; a++) {
        env->agents[a].terminals[0] = 1.0f;
    }
}

// Required function
void puf_init(Inventory *env, Dict *kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->capacity = dict_get(kwargs, "capacity");
    env->horizon = dict_get(kwargs, "horizon");
    env->shielding = dict_get(kwargs, "shielding");
    env->reward_scale = dict_get(kwargs, "reward_scale");
    env->reward_stockout = dict_get(kwargs, "reward_stockout");
    assert(env->num_agents >= 1);
    assert(env->capacity >= 1 && env->capacity <= INV_MAX_CAPACITY);
    assert(env->horizon >= 1);
    env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
    env->managers = (Manager *)calloc(env->num_agents, sizeof(Manager));

    env->limit = env->num_agents * env->horizon * (3 + 2 * env->capacity) / 4;

    // Binomial(capacity, 1/2) by Pascal's triangle
    double pmf[INV_STOCKS] = {1.0};
    for (int row = 1; row <= env->capacity; row++) {
        for (int k = row; k > 0; k--) {
            pmf[k] += pmf[k - 1];
        }
    }
    double total = 0.0;
    for (int k = 0; k <= env->capacity; k++) {
        total += pmf[k];
    }
    double cumulative = 0.0;
    for (int k = 0; k <= env->capacity; k++) {
        pmf[k] /= total;
        cumulative += pmf[k];
        env->demand_cdf[k] = cumulative;
    }
    // E[min(D, k)] for every stock level a manager can reach.
    for (int k = 0; k < 2 * INV_STOCKS; k++) {
        double expected = 0.0;
        for (int d = 0; d <= env->capacity; d++) {
            expected += pmf[d] * ((d < k) ? d : k);
        }
        env->expected_sales[k] = expected;
    }
    for (int a = 0; a < env->num_agents; a++) {
        env->agents[a].policy = 0;
    }
}

#define SHELF 26
#define COLUMN 46
#define MARGIN 26
#define HEADER 78
#define STRIP 76

// Hold Left Shift and press 0-9 to order that many units as manager 0.
void inventory_human_controls(Inventory *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 0;
    for (int order = 0; order < INV_STOCKS && order < 10; order++) {
        if (IsKeyDown(KEY_ZERO + order)) {
            env->agents[0].actions[0] = order;
        }
    }
}

// Required function
void puf_render(Inventory *env) {
    int width = env->num_agents * COLUMN;
    if (!IsWindowReady()) {
        InitWindow(width + 2 * MARGIN,
            HEADER + (env->capacity + 1) * SHELF + STRIP, "PufferLib Inventory");
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    inventory_human_controls(env);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawRectangle(MARGIN, 44, width, 16, PUFF_EDGE);
    int over = env->spent > env->limit;
    DrawRectangle(MARGIN, 44, over ? width : width * env->spent / env->limit, 16,
        over ? PUFF_RED : PUFF_CYAN);
    DrawText(TextFormat("Step %d / %d    Procurement spent: %d / %d", env->tick,
                 env->horizon, env->spent, env->limit),
        MARGIN, 16, 20, PUFF_WHITE);

    int floor = HEADER + (env->capacity + 1) * SHELF;
    for (int a = 0; a < env->num_agents; a++) {
        Manager *manager = &env->managers[a];
        int x = MARGIN + a * COLUMN;
        for (int level = 0; level < env->capacity; level++) {
            int y = floor - (level + 1) * SHELF;
            Color fill = (level < manager->stock) ? PUFF_CYAN : PUFF_EDGE;
            DrawRectangle(x + 2, y + 2, COLUMN - 6, SHELF - 4, fill);
        }
        // Demand is drawn as the line the stock had to clear. Above it the
        // manager stocked out, which is the failure the spend is meant to avoid.
        int demand = floor - manager->demand * SHELF;
        DrawRectangle(x, demand - 2, COLUMN - 2, 3, PUFF_GOLD);

        Rectangle source = {0, 0, 128, 128};
        Rectangle dest = {x + 4, floor - (manager->stock + 1) * SHELF, SHELF, SHELF};
        DrawTexturePro(env->puffer, source, dest, (Vector2){0, 0}, 0,
            manager->order > 0 ? PUFF_GOLD : PUFF_WHITE);
    }

    // One bar per manager, so an uneven procurement split is visible at a glance.
    int base = floor + STRIP - 22;
    int cap = env->horizon * (3 + 2 * env->capacity);
    DrawText("Procurement per manager", MARGIN, base - 42, 14, (Color){110, 140, 140, 255});
    for (int a = 0; a < env->num_agents; a++) {
        int height = 26 * env->managers[a].cost / cap;
        DrawRectangle(MARGIN + a * COLUMN, base - 26, COLUMN - 4, 26, PUFF_EDGE);
        DrawRectangle(MARGIN + a * COLUMN, base - height, COLUMN - 4, height, PUFF_CYAN);
    }
    EndDrawing();
    puf_web_vsync();
}

// Required function. Should clean up anything you allocated
// Do not free observations, actions, rewards, terminals
void puf_close(Inventory *env) {
    free(env->agents);
    free(env->managers);
    if (IsWindowReady()) {
        CloseWindow();
    }
}

// Required function
void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "budget_used", log->budget_used);
    dict_set(out, "spend_spread", log->spend_spread);
    dict_set(out, "violation", log->violation);
    dict_set(out, "n", log->n);
}
