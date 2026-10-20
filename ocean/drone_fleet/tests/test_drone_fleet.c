// Checks the correlated-wind construction in ocean/drone_fleet.
//
// The claim being tested is the whole point of the env: blending a shared and a
// private Ornstein-Uhlenbeck gust as sqrt(rho)*W + sqrt(1-rho)*u_i leaves each
// drone's own wind distribution untouched at every rho, while rho IS the
// cross-drone correlation. Severity and dependence move independently, which is
// what makes a rho sweep a controlled experiment.

#include "../drone_fleet.h"
#include <stdio.h>

#define CHECK(condition)                                                             \
    do {                                                                             \
        if (!(condition)) {                                                          \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
                #condition);                                                         \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CLOSE(value, want, tol) CHECK(fabsf((value) - (want)) < (tol))

#define N_DRONES 16
#define OBS_BYTES (N_DRONES * OBS_SIZE)

static obs_t observations[OBS_BYTES];
static float actions[N_DRONES * 4];
static float rewards[N_DRONES];
static float terminals[N_DRONES];

static void setup(DroneEnv *env, float sigma, float tau, float rho, float dr, unsigned seed) {
    memset(env, 0, sizeof(*env));
    env->rng = seed;
    Dict k = {0};
    dict_set(&k, "num_drones", N_DRONES);
    dict_set(&k, "hover_frac", 1.0);
    dict_set(&k, "race_frac", 0.0);
    dict_set(&k, "sphere_frac", 0.0);
    dict_set(&k, "cube_frac", 0.0);
    dict_set(&k, "flag_frac", 0.0);
    dict_set(&k, "dr", dr);
    dict_set(&k, "b_drag", 0.05);
    dict_set(&k, "use_rk2", 0);
    dict_set(&k, "alpha_vel", 0.0);
    dict_set(&k, "alpha_omega", 0.0);
    dict_set(&k, "alpha_action", 0.0);
    dict_set(&k, "oob_penalty", 0.0);
    dict_set(&k, "sphere_radius", 4.0);
    dict_set(&k, "hover_target_dist", 500.0); // no drift-termination mid-measurement
    dict_set(&k, "hover_horizon", 1024);
    dict_set(&k, "alpha_hover", 1.0);
    dict_set(&k, "hover_alpha_dist", 0.8);
    dict_set(&k, "max_rings", 10);
    dict_set(&k, "race_horizon", 2048);
    dict_set(&k, "ring_reward", 2.4);
    dict_set(&k, "race_alpha_dist", 0.8);
    dict_set(&k, "wind_sigma", sigma);
    dict_set(&k, "wind_tau", tau);
    dict_set(&k, "wind_rho", rho);
    puf_init(env, &k);
    for (int i = 0; i < N_DRONES; i++) {
        env->agents[i].observations = observations + i * OBS_SIZE;
        env->agents[i].actions = actions + i * 4;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
    }
    puf_reset(env);
}

static void hover_actions(void) {
    for (int i = 0; i < N_DRONES * 4; i++) {
        actions[i] = 0.0f;
    }
}

// Wind is off by default and the fork must then be the upstream env exactly.
static void test_zero_sigma_is_inert(void) {
    DroneEnv env;
    setup(&env, 0.0f, 1.0f, 0.5f, 0.0f, 7);
    hover_actions();
    for (int t = 0; t < 400; t++) {
        puf_step(&env);
        for (int i = 0; i < N_DRONES; i++) {
            CHECK(env.drones[i].wind.x == 0.0f);
            CHECK(env.drones[i].wind.y == 0.0f);
            CHECK(env.drones[i].wind.z == 0.0f);
        }
    }
    CHECK(env.wind_shared.x == 0.0f && env.wind_shared.y == 0.0f);
    puf_close(&env);
}

// Exact OU discretisation: the stationary std must be sigma regardless of tau.
static void test_ou_marginal(void) {
    float taus[3] = {0.1f, 0.5f, 2.0f};
    for (int j = 0; j < 3; j++) {
        DroneEnv env;
        setup(&env, 2.0f, taus[j], 0.0f, 0.0f, 11 + j);
        hover_actions();
        double sum = 0.0, sq = 0.0;
        int steps = 400000;
        for (int t = 0; t < steps; t++) {
            puf_step(&env);
            sum += env.wind_shared.x;
            sq += (double)env.wind_shared.x * env.wind_shared.x;
        }
        double mean = sum / steps;
        double std = sqrt(sq / steps - mean * mean);
        // OU sample mean has std sigma*sqrt(2*tau/T); a fixed 0.06 was ~1 SE at tau=2
        float mean_se = 2.0f * sqrtf(2.0f * taus[j] / (steps * ACTION_DT));
        CLOSE((float)mean, 0.0f, 4.0f * mean_se);
        CLOSE((float)std, 2.0f, 0.12f); // sigma, not sigma * f(tau)
        puf_close(&env);
    }
}

// Autocorrelation at lag tau must be 1/e. This is what distinguishes a gust that
// persists from white noise that averages away.
static void test_ou_autocorrelation(void) {
    DroneEnv env;
    float tau = 0.5f;
    setup(&env, 2.0f, tau, 0.0f, 0.0f, 23);
    hover_actions();
    int lag = (int)(tau / ACTION_DT); // 50 steps at 100 Hz
    int steps = 300000;
    float *trace = (float *)malloc(steps * sizeof(float));
    for (int t = 0; t < steps; t++) {
        puf_step(&env);
        trace[t] = env.wind_shared.x;
    }
    double sum = 0.0, sq = 0.0, cross = 0.0;
    int n = steps - lag;
    for (int t = 0; t < n; t++) {
        sum += trace[t];
        sq += (double)trace[t] * trace[t];
        cross += (double)trace[t] * trace[t + lag];
    }
    double mean = sum / n;
    double var = sq / n - mean * mean;
    double acf = (cross / n - mean * mean) / var;
    CLOSE((float)acf, expf(-1.0f), 0.05f);
    free(trace);
    puf_close(&env);
}

// Mean pairwise correlation of one wind component across drones, and the mean
// per-drone std, sampled over a long run.
static void wind_stats(float rho, float *corr_out, float *std_out) {
    DroneEnv env;
    setup(&env, 2.0f, 0.5f, rho, 0.0f, 101);
    hover_actions();
    int steps = 200000;
    double sum[N_DRONES] = {0}, sq[N_DRONES] = {0};
    double cross = 0.0;
    int pairs = 0;
    double *w = (double *)calloc(N_DRONES, sizeof(double));
    double pair_sum = 0.0;
    for (int t = 0; t < steps; t++) {
        puf_step(&env);
        for (int i = 0; i < N_DRONES; i++) {
            w[i] = env.drones[i].wind.x;
            sum[i] += w[i];
            sq[i] += w[i] * w[i];
        }
        for (int i = 0; i < N_DRONES; i++) {
            for (int j = i + 1; j < N_DRONES; j++) {
                pair_sum += w[i] * w[j];
            }
        }
    }
    pairs = N_DRONES * (N_DRONES - 1) / 2;
    cross = pair_sum / ((double)steps * pairs);

    double mean_var = 0.0, mean_std = 0.0;
    for (int i = 0; i < N_DRONES; i++) {
        double m = sum[i] / steps;
        double v = sq[i] / steps - m * m;
        mean_var += v;
        mean_std += sqrt(v);
    }
    mean_var /= N_DRONES;
    mean_std /= N_DRONES;
    *corr_out = (float)(cross / mean_var); // means are ~0
    *std_out = (float)mean_std;
    free(w);
    puf_close(&env);
}

// The claim of the whole construction: rho is the cross-drone correlation, and
// the per-drone marginal does not move with it.
static void test_rho_is_the_correlation(void) {
    float rhos[5] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    printf("  rho    measured corr    per-drone std (sigma = 2.0)\n");
    for (int j = 0; j < 5; j++) {
        float corr, std;
        wind_stats(rhos[j], &corr, &std);
        printf("  %.2f   %12.4f    %.4f\n", rhos[j], corr, std);
        CLOSE(corr, rhos[j], 0.03f);
        CLOSE(std, 2.0f, 0.10f); // severity is invariant to the dependence dial
    }
}

// Wind must actually reach the dynamics, and its correlation must survive the
// trip. Open loop (constant action) the drones simply drift, so displacement is a
// clean readout of the disturbance itself.
static void test_wind_reaches_the_dynamics(void) {
    float rhos[3] = {0.0f, 0.5f, 1.0f};
    printf("  rho    drift corr    mean |drift| (m)\n");
    for (int j = 0; j < 3; j++) {
        int trials = 600;
        int window = 100; // 1 s
        double sum[N_DRONES] = {0}, sq[N_DRONES] = {0};
        double pair_sum = 0.0, mag = 0.0;
        for (int tr = 0; tr < trials; tr++) {
            DroneEnv env;
            setup(&env, 2.0f, 0.5f, rhos[j], 0.0f, 9000 + tr);
            hover_actions();
            // Pin every drone, in every trial, to one canonical level-hover
            // state. This is a measurement device local to this test, NOT a
            // change to the env: puf_reset still randomises spawn pose exactly as
            // upstream, and training sees the full distribution. Pinning here
            // removes two confounds that have nothing to do with the claim under
            // test - per-drone pose spread, and a per-trial common tilt shared by
            // the whole fleet. Level attitude puts thrust purely vertical, so
            // lateral drift is wind and nothing else.
            for (int i = 0; i < N_DRONES; i++) {
                env.drones[i].params = env.drones[0].params;
                State st = {0};
                st.quat.w = 1.0f;
                env.drones[i].state = st;
                physics_set_drone((Physics *)env.physics, i, &env.drones[i].params, &st);
            }
            Vec3 start[N_DRONES];
            for (int i = 0; i < N_DRONES; i++) {
                start[i] = env.drones[i].state.pos;
            }
            for (int t = 0; t < window; t++) {
                puf_step(&env);
            }
            // A reset mid-window teleports the drone and would silently poison the
            // displacement. Assert it did not happen rather than trusting it.
            for (int i = 0; i < N_DRONES; i++) {
                CHECK(env.drones[i].episode_length == window);
            }
            double d[N_DRONES];
            for (int i = 0; i < N_DRONES; i++) {
                d[i] = env.drones[i].state.pos.x - start[i].x;
                sum[i] += d[i];
                sq[i] += d[i] * d[i];
                mag += fabs(d[i]);
            }
            for (int i = 0; i < N_DRONES; i++) {
                for (int k = i + 1; k < N_DRONES; k++) {
                    pair_sum += d[i] * d[k];
                }
            }
            puf_close(&env);
        }
        int pairs = N_DRONES * (N_DRONES - 1) / 2;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < N_DRONES; i++) {
            double m = sum[i] / trials;
            mean += m;
            var += sq[i] / trials - m * m;
        }
        mean /= N_DRONES;
        var /= N_DRONES;
        double corr = (pair_sum / ((double)trials * pairs) - mean * mean) / var;
        printf("  %.2f   %10.4f    %.4f\n", rhos[j], corr, mag / (trials * N_DRONES));
        CHECK(var > 1e-6); // wind genuinely moves the drones
        CLOSE((float)corr, rhos[j], 0.06f);
    }
}

// Energy is NOT a clean readout, and it is worth pinning down why. rpm tracks the
// commanded rpm, which comes from the action alone, so under a constant action the
// thrust integral is invariant to wind by construction. Wind reaches the cost only
// through a policy's response to being pushed off station. This asserts the
// invariance rather than hiding it, because it decides how the shared-energy task
// has to be built: an open-loop policy pays no wind penalty at all.
static void test_energy_is_action_determined_open_loop(void) {
    float e[2];
    for (int j = 0; j < 2; j++) {
        DroneEnv env;
        setup(&env, j == 0 ? 0.0f : 4.0f, 0.5f, 1.0f, 0.0f, 31);
        hover_actions();
        for (int t = 0; t < 200; t++) {
            puf_step(&env);
        }
        e[j] = env.drones[0].energy;
        puf_close(&env);
    }
    CLOSE(e[0], e[1], 1e-6f * fmaxf(e[0], 1.0f));
    printf("  open loop: energy with no wind %.6f, with sigma=4 wind %.6f (identical)\n",
        e[0], e[1]);
}

// Tab calls task_swap. It reallocates task config and state every time, so it has
// to survive repeated cycling without leaking or leaving a task half-built.
static void test_task_swap_cycles_all_tasks(void) {
    DroneEnv env;
    setup(&env, 0.0f, 1.0f, 0.0f, 0.0f, 77);
    hover_actions();
    const char *names[NUM_TASKS] = {"hover", "race", "sphere", "cube", "flag"};
    for (int lap = 0; lap < 4; lap++) {
        for (int t = 0; t < NUM_TASKS; t++) {
            task_swap(&env, (TaskType)t);
            CHECK(env.task == (TaskType)t);
            CHECK(env.task_config != NULL);
            CHECK(env.task_state != NULL);
            // and it must still run afterwards
            for (int k = 0; k < 20; k++) {
                puf_step(&env);
            }
            for (int i = 0; i < N_DRONES; i++) {
                CHECK(isfinite(env.drones[i].state.pos.x));
                CHECK(isfinite(env.drones[i].state.vel.x));
            }
            if (lap == 0) {
                printf("  task_swap -> %-7s config ok, 20 steps ok\n", names[t]);
            }
        }
    }
    puf_close(&env);
}

int main(void) {
    test_zero_sigma_is_inert();
    test_ou_marginal();
    test_ou_autocorrelation();
    test_rho_is_the_correlation();
    test_wind_reaches_the_dynamics();
    test_energy_is_action_determined_open_loop();
    test_task_swap_cycles_all_tasks();
    printf("All drone_fleet wind tests passed!\n");
    return 0;
}
