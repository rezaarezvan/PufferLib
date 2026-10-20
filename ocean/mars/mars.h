#include "raylib.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"

#ifndef MARS_MAX_GRID
#define MARS_MAX_GRID 8
#endif
#if MARS_MAX_GRID < 2 || MARS_MAX_GRID > 64
#error "MARS_MAX_GRID must be in [2, 64]"
#endif
#define MARS_CELLS (MARS_MAX_GRID * MARS_MAX_GRID)

#define ACT_SIZES {10}
#define OBS_SIZE (3 * MARS_CELLS + 5)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 6

// Actions 0-3 move regular, 4-7 move safe, in the DIRECTIONS order below.
const int SAFE = 4;
const int ACT = 8;
const int WAIT = 9;

const int DX[4] = {0, 0, -1, 1};
const int DY[4] = {1, -1, 0, 0};

// A regular move lands as intended with 0.4, on each of the three other neighbours with 0.1, stays with 0.1, and ends the rover with 0.2.
const float REGULAR_INTENDED = 0.4f;
const float REGULAR_OTHER = 0.1f;
const float REGULAR_STAY = 0.1f;
const float REGULAR_FAIL = 0.2f;

// Safe and wait end the rover with 0.05. Blocked moves fold
const float SAFE_FAIL = 0.05f;
const float WAIT_FAIL = 0.05f;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Required struct. Only use floats!
struct Log {
  float perf; // fraction of rovers that completed a task
  float score;
  float episode_return;
  float episode_length;
  float budget_used; // team spend as a fraction of the limit, over 1 if overspent
  float safe_move_rate;
  float violation; // Pr(C > L)
  float n; // Required as the last field
};

typedef struct Layout Layout;
struct Layout {
  unsigned char walls[MARS_CELLS];     // 1 where cell y*size + x is an obstacle
  unsigned char task_rank[MARS_CELLS]; // 0 where there is no task, else reward
};

typedef struct Rover Rover;
struct Rover {
  int layout; // index into the bank; redrawn every episode
  unsigned char
      visited[MARS_CELLS]; // entered this episode, for the new-cell reward
  int x;
  int y;
  int done;
  int cost;
  float ret;
};

// Required that you have some struct for your env
struct Env {
  Log log; // Required field. Env binding code uses this to aggregate logs
  Agent *agents;
  int tag;
  int boundary_reached;
  int num_agents;
  Rover *rovers;
  Layout *layouts;
  int num_layouts;
  float reward_task;
  float reward_new_cell;
  float reward_lost;
  int shielding;
  int size;
  int horizon;
  int base_seed;
  int max_rank;
  int tick;
  int spent;
  int limit;
  int safe_moves;
  Texture2D puffer;
  unsigned int rng;
};
typedef Env Mars;

int number_of_reachable_cells(unsigned char *walls, int size) {
  int queue[MARS_CELLS];
  unsigned char seen[MARS_CELLS] = {1};
  queue[0] = 0;
  int head = 0;
  int tail = 1;
  while (head < tail) {
    int cell = queue[head++];
    int x = cell % size;
    int y = cell / size;
    for (int d = 0; d < 4; d++) {
      int nx = x + DX[d];
      int ny = y + DY[d];
      if (nx < 0 || nx >= size || ny < 0 || ny >= size) {
        continue;
      }
      int next = ny * size + nx;
      if (walls[next] || seen[next]) {
        continue;
      }
      seen[next] = 1;
      queue[tail++] = next;
    }
  }
  return tail;
}

// Samples int(0.4 size^2) walls and ceil(0.1 size^2) tasks, resampling the
// walls until more than half the grid is reachable from the start. Task reward
// is the rank by ascending Manhattan distance, so the far tasks are worth the
// most.
void generate_layout(Layout *layout, int size, unsigned int seed) {
  int cells = size * size;
  int num_walls = cells * 4 / 10;
  int num_tasks = (cells + 9) / 10;
  unsigned int rng = seed;
  int budget[MARS_CELLS];
  int count = 0;
  for (int cell = 1; cell < cells; cell++) {
    budget[count++] = cell;
  }

  for (int attempt = 0; attempt < 1000; attempt++) {
    memset(layout->walls, 0, MARS_CELLS);
    for (int i = 0; i < num_walls; i++) {
      int j = i + rand_r(&rng) % (count - i);
      int swap = budget[i];
      budget[i] = budget[j];
      budget[j] = swap;
      layout->walls[budget[i]] = 1;
    }
    if (2 * number_of_reachable_cells(layout->walls, size) > cells) {
      break;
    }
  }

  count = 0;
  for (int cell = 1; cell < cells; cell++) {
    if (!layout->walls[cell]) {
      budget[count++] = cell;
    }
  }
  for (int i = 0; i < num_tasks; i++) {
    int j = i + rand_r(&rng) % (count - i);
    int swap = budget[i];
    budget[i] = budget[j];
    budget[j] = swap;
  }
  for (int i = 1; i < num_tasks; i++) {
    int cell = budget[i];
    int key = cell % size + cell / size;
    int j = i - 1;
    while (j >= 0 && budget[j] % size + budget[j] / size > key) {
      budget[j + 1] = budget[j];
      j--;
    }
    budget[j + 1] = cell;
  }
  memset(layout->task_rank, 0, MARS_CELLS);
  for (int i = 0; i < num_tasks; i++) {
    layout->task_rank[budget[i]] = i + 1;
  }
}

void compute_observations(Mars *env) {
  for (int a = 0; a < env->num_agents; a++) {
    Rover *rover = &env->rovers[a];
    Layout *layout = &env->layouts[rover->layout];
    obs_t *obs = env->agents[a].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    for (int cell = 0; cell < env->size * env->size; cell++) {
      obs[cell] = layout->walls[cell];
      obs[MARS_CELLS + cell] =
          (float)layout->task_rank[cell] / (float)env->max_rank;
    }
    int cell = rover->y * env->size + rover->x;
    obs[2 * MARS_CELLS + cell] = 1.0f;
    obs[3 * MARS_CELLS + 0] = rover->done;
    obs[3 * MARS_CELLS + 1] = (float)env->tick / (float)env->horizon;
    obs[3 * MARS_CELLS + 2] = (float)rover->cost / (float)env->horizon;
    obs[3 * MARS_CELLS + 3] =
        (float)(env->limit - env->spent) / (float)env->limit;
    obs[3 * MARS_CELLS + 4] =
        (env->num_agents > 1) ? (float)a / (env->num_agents - 1) : 0.0f;

    // A finished rover and a walled cell both only ever wait.
    // Act needs a task under the rover.
    unsigned char *mask = env->agents[a].action_mask;
    int stuck = rover->done || layout->walls[cell];
    memset(mask, !stuck, 10);
    if (env->shielding && env->spent >= env->limit) {
      memset(mask + SAFE, 0, 4);
    }
    mask[ACT] = !stuck && layout->task_rank[cell] > 0;
    mask[WAIT] = 1;
  }
}

// Required function
void puf_reset(Mars *env) {
  env->tick = 0;
  env->spent = 0;
  env->safe_moves = 0;
  for (int a = 0; a < env->num_agents; a++) {
    env->rovers[a] = (Rover){.layout = rand_r(&env->rng) % env->num_layouts};
    env->rovers[a].visited[0] = 1;
  }
  compute_observations(env);
}

void add_log(Mars *env) {
  float agents = env->num_agents;
  float ret = 0.0f;
  float tasks = 0.0f;
  for (int a = 0; a < env->num_agents; a++) {
    ret += env->rovers[a].ret;
    tasks += env->rovers[a].ret > 0.0f;
  }
  env->log.perf += tasks / agents;
  env->log.score += ret;
  env->log.episode_return += ret / agents;
  env->log.episode_length += env->horizon;
  env->log.budget_used += (float)env->spent / (float)env->limit;
  env->log.safe_move_rate += env->safe_moves / (agents * env->horizon);
  env->log.violation += env->spent > env->limit;
  env->log.n++;
}

// Required function
void puf_step(Mars *env) {
  for (int a = 0; a < env->num_agents; a++) {
    env->agents[a].rewards[0] = 0.0f;
    env->agents[a].terminals[0] = 0.0f;
  }
  env->tick++;

  for (int a = 0; a < env->num_agents; a++) {
    Rover *rover = &env->rovers[a];
    Layout *layout = &env->layouts[rover->layout];
    int cell = rover->y * env->size + rover->x;
    if (rover->done || layout->walls[cell]) {
      continue;
    }
    int action = env->agents[a].actions[0];
    if (action == ACT && layout->task_rank[cell]) {
      // ret stays the raw rank: score reports the domain objective, unshaped.
      env->agents[a].rewards[0] = env->reward_task * layout->task_rank[cell];
      rover->ret += layout->task_rank[cell];
      rover->done = 1;
      continue;
    }

    float draw = (float)rand_r(&env->rng) / (float)RAND_MAX;
    if (action == ACT || action == WAIT) {
      rover->done = draw < WAIT_FAIL;
      env->agents[a].rewards[0] += rover->done * env->reward_lost;
      continue;
    }
    int direction = action;
    if (action >= SAFE) {
      direction = action - SAFE;
      // Charged in agent order, so a rover reaching for the last unit in the
      // same step as an earlier one finds it gone and holds position.
      if (env->shielding && env->spent == env->limit) {
        rover->done = draw < WAIT_FAIL;
        env->agents[a].rewards[0] += rover->done * env->reward_lost;
        continue;
      }
      rover->cost++;
      env->spent++;
      env->safe_moves++;
      if (draw < SAFE_FAIL) {
        rover->done = 1;
        env->agents[a].rewards[0] += env->reward_lost;
        continue;
      }
    } else if (draw < REGULAR_FAIL) {
      rover->done = 1;
      env->agents[a].rewards[0] += env->reward_lost;
      continue;
    } else if (draw < REGULAR_FAIL + REGULAR_STAY) {
      continue;
    } else if (draw >= REGULAR_FAIL + REGULAR_STAY + REGULAR_INTENDED) {
      int offset = (draw - REGULAR_FAIL - REGULAR_STAY - REGULAR_INTENDED) /
                   REGULAR_OTHER;
      direction = (action + 1 + (offset < 3 ? offset : 2)) % 4;
    }

    int x = rover->x + DX[direction];
    int y = rover->y + DY[direction];
    if (x < 0 || x >= env->size || y < 0 || y >= env->size) {
      continue;
    }
    if (layout->walls[y * env->size + x]) {
      continue;
    }
    rover->x = x;
    rover->y = y;

    // Paying for first visits pushes rovers outward toward the far, higher-rank
    // tasks instead of settling for the nearest one. Same shape as
    // ocean/pathfinder's new_cell_reward.
    if (!rover->visited[y * env->size + x]) {
      rover->visited[y * env->size + x] = 1;
      env->agents[a].rewards[0] += env->reward_new_cell;
    }
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
void puf_init(Mars *env, Dict *kwargs) {
  env->num_agents = dict_get(kwargs, "num_agents");
  env->size = dict_get(kwargs, "size");
  env->base_seed = dict_get(kwargs, "base_seed");
  env->num_layouts = dict_get(kwargs, "num_layouts");
  env->shielding = dict_get(kwargs, "shielding");
  env->reward_task = dict_get(kwargs, "reward_task");
  env->reward_new_cell = dict_get(kwargs, "reward_new_cell");
  env->reward_lost = dict_get(kwargs, "reward_lost");
  assert(env->num_agents >= 1);
  assert(env->size >= 2 && env->size <= MARS_MAX_GRID);
  assert(env->num_layouts >= 1);
  env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
  env->rovers = (Rover *)calloc(env->num_agents, sizeof(Rover));
  env->layouts = (Layout *)calloc(env->num_layouts, sizeof(Layout));

  env->horizon = 2 * env->size;
  env->limit = env->horizon * env->num_agents / 4;
  env->max_rank = (env->size * env->size + 9) / 10;
  for (int j = 0; j < env->num_layouts; j++) {
    generate_layout(&env->layouts[j], env->size, env->base_seed + j);
  }
  for (int a = 0; a < env->num_agents; a++) {
    env->agents[a].policy = 0;
  }
}

// Hold Left Shift + WASD/arrows to drive rover 0. Tab toggles safe moves.
void mars_human_controls(Mars *env) {
  if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
    return;
  }
  int safe = IsKeyDown(KEY_TAB) ? SAFE : 0;
  env->agents[0].actions[0] = WAIT;
  if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
    env->agents[0].actions[0] = safe + 0;
  } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
    env->agents[0].actions[0] = safe + 1;
  } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
    env->agents[0].actions[0] = safe + 2;
  } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
    env->agents[0].actions[0] = safe + 3;
  } else if (IsKeyDown(KEY_SPACE)) {
    env->agents[0].actions[0] = ACT;
  }
}

// Required function
void puf_render(Mars *env) {
  int tile = 40;
  int header = 64;
  int gutter = 16;
  int cols = 1;
  while (cols * cols < env->num_agents) {
    cols++;
  }
  int rows = (env->num_agents + cols - 1) / cols;
  int span = env->size * tile + gutter;
  if (!IsWindowReady()) {
    InitWindow(cols * span + gutter, rows * span + gutter + header,
               "PufferLib Mars");
    env->puffer = LoadTexture("resources/shared/puffers_128.png");
  }
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }
  mars_human_controls(env);

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);
  int width = cols * span - gutter;
  DrawRectangle(gutter, 36, width, 14, (Color){20, 40, 40, 255});
  int over = env->spent > env->limit;
  DrawRectangle(gutter, 36, over ? width : width * env->spent / env->limit, 14,
                over ? PUFF_RED : PUFF_CYAN);
  DrawText(TextFormat("Step %d / %d    Budget: %d / %d", env->tick,
                      env->horizon, env->spent, env->limit),
           gutter, 12, 18, PUFF_WHITE);

  for (int a = 0; a < env->num_agents; a++) {
    Rover *rover = &env->rovers[a];
    Layout *layout = &env->layouts[rover->layout];
    int ox = gutter + (a % cols) * span;
    int oy = header + gutter + (a / cols) * span;
    for (int cell = 0; cell < env->size * env->size; cell++) {
      // y grows upward in the domain, so flip it for the screen.
      int x = ox + (cell % env->size) * tile;
      int y = oy + (env->size - 1 - cell / env->size) * tile;
      if (layout->walls[cell]) {
        DrawRectangle(x, y, tile, tile, (Color){40, 48, 48, 255});
      } else if (layout->task_rank[cell]) {
        int shade = 60 + 160 * layout->task_rank[cell] / env->max_rank;
        DrawRectangle(x, y, tile, tile, (Color){shade, 0, 0, 255});
        DrawText(TextFormat("%d", layout->task_rank[cell]), x + 4, y + 4, 14,
                 PUFF_WHITE);
      }
      DrawRectangleLines(x, y, tile, tile, (Color){20, 40, 40, 255});
    }
    Rectangle source = {0, 0, 128, 128};
    Rectangle dest = {ox + rover->x * tile,
                      oy + (env->size - 1 - rover->y) * tile, tile, tile};
    DrawTexturePro(env->puffer, source, dest, (Vector2){0, 0}, 0,
                   rover->done ? (Color){80, 80, 80, 160} : WHITE);
  }
  EndDrawing();
  puf_web_vsync();
}

// Required function. Should clean up anything you allocated
// Do not free observations, actions, rewards, terminals
void puf_close(Mars *env) {
  free(env->agents);
  free(env->rovers);
  free(env->layouts);
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
  dict_set(out, "safe_move_rate", log->safe_move_rate);
  dict_set(out, "violation", log->violation);
  dict_set(out, "n", log->n);
}
