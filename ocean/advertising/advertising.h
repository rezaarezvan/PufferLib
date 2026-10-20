#include "raylib.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"

#define AD_STATES 16
#define AD_STRATEGIES 5

#define ACT_SIZES {7}
#define OBS_SIZE (AD_STATES + 4)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 6

// Actions 0-4 are the ad strategies, indexed by their own cost.
const int BUY = 5;
const int END = 6;

const int LOST = 14;
const int READY = 15;

// A costlier strategy advances the customer more often and loses them less.
// CONT_AB and CONT_AC split the fork at state 5 between the two downstream branches.
// REGRESS_C pulls the customer back out of branch C.
const float CONT_A[AD_STRATEGIES] = {0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
const float CONT_AB[AD_STRATEGIES] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
const float CONT_AC[AD_STRATEGIES] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
const float REGRESS_C[AD_STRATEGIES] = {0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

// What a purchase is worth in the paper, score is reported in these units.
const float PURCHASE_REWARD = 100.0f;

// Progress along the funnel.
// Branch C counts down because it is a detour.
// Reward is paid on the change in depth.
const int DEPTH[AD_STATES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3, 2, 0, 10};

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_GOLD = (Color){235, 185, 70, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_EDGE = (Color){20, 52, 52, 255};

// Cheap campaigns are dim and cool, expensive ones hot.
// Tints a puffer sprite, so this is the team's spending pattern at a glance.
const Color STRATEGY_TINT[AD_STRATEGIES] = {
    (Color){45, 92, 92, 255},
    (Color){0, 140, 150, 255},
    (Color){0, 190, 190, 255},
    (Color){225, 150, 55, 255},
    (Color){235, 85, 60, 255},
};

// Required struct. Only use floats!
struct Log {
    float perf; // fraction of advertisers whose customer bought
    float score;
    float episode_return;
    float episode_length;
    float budget_used;  // team spend as a fraction of the limit, over 1 if overspent
    float spend_spread; // widest gap between two advertisers, over the fair share
    float violation;    // Pr(C > L)
    float n;            // Required as the last field
};

typedef struct Advertiser Advertiser;
struct Advertiser {
    int state;
    int cost;
    int strategy; // what it actually ran last step, for the render
    int bought;
};

// Required that you have some struct for your env
struct Env {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Agent *agents;
    int tag;
    int boundary_reached;
    int num_agents;
    Advertiser *advertisers;
    int shielding;
    int horizon;
    int tick;
    int spent;
    int limit;
    float reward_purchase;
    float reward_progress;
    float reward_lost;
    Texture2D puffer;
    unsigned int rng;
};
typedef Env Advertising;

int funnel_outcomes(int state, int strategy, int *next, float *probability) {
    float half = (1.0f - CONT_A[strategy]) / 2.0f;
    float third = (1.0f - CONT_A[strategy]) / 3.0f;
    float cthird = (1.0f - REGRESS_C[strategy]) / 3.0f;
    if (state < 5) {
        next[0] = state + 1;
        next[1] = LOST;
        next[2] = state;
        probability[0] = CONT_A[strategy];
        probability[1] = half;
        probability[2] = half;
        return 3;
    }
    if (state == 5) {
        next[0] = 6;
        next[1] = 10;
        next[2] = LOST;
        next[3] = 5;
        probability[0] = CONT_AB[strategy];
        probability[1] = CONT_AC[strategy];
        probability[2] = half;
        probability[3] = half;
        return 4;
    }
    if (state < 10) {
        next[0] = (state == 9) ? READY : state + 1;
        next[1] = state - 1;
        next[2] = LOST;
        next[3] = state;
        probability[0] = CONT_A[strategy];
        probability[1] = third;
        probability[2] = third;
        probability[3] = third;
        return 4;
    }
    if (state == 10) {
        next[0] = 11;
        next[1] = 5;
        next[2] = LOST;
        next[3] = 10;
        probability[0] = cthird;
        probability[1] = REGRESS_C[strategy];
        probability[2] = cthird;
        probability[3] = cthird;
        return 4;
    }
    next[0] = state - 1;
    next[1] = LOST;
    next[2] = state;
    next[3] = state + 1;
    probability[0] = REGRESS_C[strategy];
    probability[1] = cthird;
    probability[2] = cthird;
    probability[3] = cthird;
    return 4;
}

void compute_observations(Advertising *env) {
    int remaining = env->limit - env->spent;
    for (int a = 0; a < env->num_agents; a++) {
        Advertiser *advertiser = &env->advertisers[a];
        obs_t *obs = env->agents[a].observations;
        memset(obs, 0, OBS_SIZE * sizeof(obs_t));
        obs[advertiser->state] = 1.0f;
        obs[AD_STATES + 0] = (float)env->tick / (float)env->horizon;
        obs[AD_STATES + 1] = (float)advertiser->cost / (float)env->limit;
        obs[AD_STATES + 2] = (float)remaining / (float)env->limit;
        obs[AD_STATES + 3] =
            (env->num_agents > 1) ? (float)a / (env->num_agents - 1) : 0.0f;

        unsigned char *mask = env->agents[a].action_mask;
        int running = advertiser->state < LOST;
        for (int strategy = 0; strategy < AD_STRATEGIES; strategy++) {
            mask[strategy] = running && (!env->shielding || strategy <= remaining);
        }
        mask[BUY] = advertiser->state == READY;
        mask[END] = advertiser->state == LOST;
    }
}

// Required function
void puf_reset(Advertising *env) {
    env->tick = 0;
    env->spent = 0;
    for (int a = 0; a < env->num_agents; a++) {
        env->advertisers[a] = (Advertiser){0};
    }
    compute_observations(env);
}

void add_log(Advertising *env) {
    float agents = env->num_agents;
    float purchases = 0.0f;
    int most = 0;
    int least = env->limit;
    for (int a = 0; a < env->num_agents; a++) {
        Advertiser *advertiser = &env->advertisers[a];
        purchases += advertiser->bought;
        most = (advertiser->cost > most) ? advertiser->cost : most;
        least = (advertiser->cost < least) ? advertiser->cost : least;
    }
    env->log.perf += purchases / agents;
    env->log.score += PURCHASE_REWARD * purchases;
    env->log.episode_return += purchases / agents;
    env->log.episode_length += env->horizon;
    env->log.budget_used += (float)env->spent / (float)env->limit;
    env->log.spend_spread += (float)(most - least) * agents / (float)env->limit;
    env->log.violation += env->spent > env->limit;
    env->log.n++;
}

// Required function
void puf_step(Advertising *env) {
    for (int a = 0; a < env->num_agents; a++) {
        env->agents[a].rewards[0] = 0.0f;
        env->agents[a].terminals[0] = 0.0f;
    }
    env->tick++;

    for (int a = 0; a < env->num_agents; a++) {
        Advertiser *advertiser = &env->advertisers[a];
        int action = env->agents[a].actions[0];
        if (advertiser->state == LOST) {
            continue;
        }
        if (advertiser->state == READY) {
            if (action == BUY) {
                env->agents[a].rewards[0] = env->reward_purchase;
                advertiser->bought = 1;
                advertiser->state = LOST;
            }
            continue;
        }

        int strategy = (action < AD_STRATEGIES) ? action : 0;
        if (env->shielding && strategy > env->limit - env->spent) {
            strategy = env->limit - env->spent;
        }
        advertiser->cost += strategy;
        advertiser->strategy = strategy;
        env->spent += strategy;

        int next[4];
        float probability[4];
        int from = advertiser->state;
        int outcomes = funnel_outcomes(from, strategy, next, probability);
        float draw = (float)rand_r(&env->rng) / (float)RAND_MAX;
        float cumulative = 0.0f;
        advertiser->state = next[outcomes - 1];
        for (int i = 0; i < outcomes - 1; i++) {
            cumulative += probability[i];
            if (draw < cumulative) {
                advertiser->state = next[i];
                break;
            }
        }

        if (advertiser->state == LOST) {
            env->agents[a].rewards[0] += env->reward_lost;
            continue;
        }
        env->agents[a].rewards[0] +=
            env->reward_progress * (DEPTH[advertiser->state] - DEPTH[from]);
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
void puf_init(Advertising *env, Dict *kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->horizon = dict_get(kwargs, "horizon");
    env->shielding = dict_get(kwargs, "shielding");
    assert(env->num_agents >= 1);
    assert(env->horizon >= 1);
    env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
    env->advertisers = (Advertiser *)calloc(env->num_agents, sizeof(Advertiser));

    env->reward_purchase = dict_get(kwargs, "reward_purchase");
    env->reward_progress = dict_get(kwargs, "reward_progress");
    env->reward_lost = dict_get(kwargs, "reward_lost");

    env->limit = 10 * env->num_agents;
    for (int a = 0; a < env->num_agents; a++) {
        env->agents[a].policy = 0;
    }
}

const int NODE_COLUMN[AD_STATES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 0, 10};
const int NODE_ROW[AD_STATES] = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 3, 0};
const int EDGE_FROM[14] = {0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12};
const int EDGE_TO[14] = {1, 2, 3, 4, 5, 6, 10, 7, 8, 9, 15, 11, 12, 13};

#define NODE 54
#define PITCH 90
#define MARGIN 26
#define HEADER 78
#define STRIP 76
#define COLUMNS 11
#define BOARD (MARGIN + (COLUMNS - 1) * PITCH + NODE)

void advertising_human_controls(Advertising *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 0;
    for (int strategy = 0; strategy < AD_STRATEGIES; strategy++) {
        if (IsKeyDown(KEY_ONE + strategy)) {
            env->agents[0].actions[0] = strategy;
        }
    }
    if (IsKeyDown(KEY_SPACE)) {
        env->agents[0].actions[0] = BUY;
    }
}

// Required function
void puf_render(Advertising *env) {
    if (!IsWindowReady()) {
        InitWindow(BOARD + MARGIN, HEADER + 3 * PITCH + NODE + STRIP,
            "PufferLib Advertising");
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    advertising_human_controls(env);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    int width = BOARD - MARGIN;
    DrawRectangle(MARGIN, 44, width, 16, PUFF_EDGE);
    int over = env->spent > env->limit;
    DrawRectangle(MARGIN, 44, over ? width : width * env->spent / env->limit, 16,
        over ? PUFF_RED : STRATEGY_TINT[4]);
    DrawText(TextFormat("Step: %d / %d    Budget: %d / %d", env->tick,
                 env->horizon, env->spent, env->limit),
        MARGIN, 16, 20, PUFF_WHITE);

    for (int e = 0; e < 14; e++) {
        int from = EDGE_FROM[e];
        int to = EDGE_TO[e];
        Vector2 start = {MARGIN + NODE_COLUMN[from] * PITCH + NODE,
            HEADER + NODE_ROW[from] * PITCH + NODE / 2.0f};
        Vector2 end = {MARGIN + NODE_COLUMN[to] * PITCH,
            HEADER + NODE_ROW[to] * PITCH + NODE / 2.0f};
        DrawLineEx(start, end, 3, PUFF_EDGE);
    }

    for (int state = 0; state < AD_STATES; state++) {
        Rectangle box = {MARGIN + NODE_COLUMN[state] * PITCH,
            HEADER + NODE_ROW[state] * PITCH, NODE, NODE};
        Color fill = PUFF_EDGE;
        const char *label = TextFormat("%d", state);
        if (state == LOST) {
            box.width = width;
            fill = (Color){44, 30, 30, 255};
            label = "LOST";
        } else if (state == READY) {
            fill = (Color){80, 62, 20, 255};
            label = "BUY";
        }
        DrawRectangleRec(box, fill);
        DrawRectangleLinesEx(box, 2, PUFF_BACKGROUND);
        DrawText(label, box.x + 5, box.y + 4, 14,
            (state == READY) ? PUFF_GOLD : (Color){110, 140, 140, 255});

        int occupants = 0;
        for (int a = 0; a < env->num_agents; a++) {
            occupants += env->advertisers[a].state == state;
        }
        int columns = 1;
        while (columns * ((columns * NODE) / (int)box.width + 1) < occupants) {
            columns++;
        }
        int rows = (occupants + columns - 1) / columns;
        float sprite = box.width / columns;
        if (box.height / rows < sprite) {
            sprite = box.height / rows;
        }

        int slot = 0;
        for (int a = 0; a < env->num_agents; a++) {
            Advertiser *advertiser = &env->advertisers[a];
            if (advertiser->state != state) {
                continue;
            }
            Rectangle source = {0, 0, 128, 128};
            Rectangle dest = {box.x + (slot % columns) * sprite,
                box.y + (slot / columns) * sprite, sprite, sprite};
            Color tint = STRATEGY_TINT[advertiser->strategy];
            if (advertiser->bought) {
                tint = PUFF_GOLD;
            } else if (state == LOST) {
                tint = (Color){90, 90, 90, 190};
            }
            DrawTexturePro(env->puffer, source, dest, (Vector2){0, 0}, 0, tint);
            slot++;
        }
    }

    int base = HEADER + 3 * PITCH + NODE + STRIP - 22;
    float bar = (float)width / env->num_agents;
    DrawText("Spend per advertiser", MARGIN, base - 42, 14, (Color){110, 140, 140, 255});
    for (int a = 0; a < env->num_agents; a++) {
        Advertiser *advertiser = &env->advertisers[a];
        int height = 26 * advertiser->cost / (4 * env->horizon);
        Color tint = STRATEGY_TINT[4];
        if (advertiser->bought) {
            tint = PUFF_GOLD;
        } else if (advertiser->state == LOST) {
            tint = (Color){90, 90, 90, 255};
        }
        DrawRectangle(MARGIN + a * bar, base - 26, bar - 2, 26, PUFF_EDGE);
        DrawRectangle(MARGIN + a * bar, base - height, bar - 2, height, tint);
    }
    EndDrawing();
    puf_web_vsync();
}

// Required function. Should clean up anything you allocated
// Do not free observations, actions, rewards, terminals
void puf_close(Advertising *env) {
    free(env->agents);
    free(env->advertisers);
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
