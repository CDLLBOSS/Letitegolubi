"""
ROB-UY 3303 - Lab 03: Path Following
STARTER CODE - complete every function marked STUDENT TODO.
Run the unit tests with:   python lab_3.py
The GUI (gui_03.py) imports this file and uses your controllers on the robot.

Write your code only where it says STUDENT TODO. Sections marked
PROVIDED are given to you: do not change them.

Units: meters, seconds, radians.  v = forward speed (m/s), w = rotational
speed (rad/s, + = counter-clockwise), phi_dot_l / phi_dot_r = wheel speeds (rad/s).
"""
import math

# =====================================================================
# STUDENT TODO - ROBOT CONSTANTS (your Lab 2 values, same as in gui_03.py)
# =====================================================================
WHEEL_RADIUS_M = 0.067 / 2   # Lab 2: 0.0335 m
WHEELBASE_M = 0.177        # Lab 2: 2L in meters


# =====================================================================
# PART 1 - WHEEL VELOCITIES
# =====================================================================
# STUDENT TODO: get_wheel_velocities(v, w) -> (phi_dot_l, phi_dot_r)
# Use the equations you derived in Step 1.
def get_wheel_velocities(v, w):
    # Inverse differential-drive kinematics: v = r*(phi_r+phi_l)/2,
    # w = r*(phi_r-phi_l)/(2*L), with L = half the wheelbase.
    L = WHEELBASE_M / 2.0
    phi_dot_l = (v - L * w) / WHEEL_RADIUS_M
    phi_dot_r = (v + L * w) / WHEEL_RADIUS_M
    return phi_dot_l, phi_dot_r


# STUDENT TODO: 5 unit tests, each returns True if it passes.
#   1: v > 0, w = 0     2: v < 0, w = 0     3: v = 0, w > 0
#   4: v = 0, w < 0     5: v = 0, w = 0
TOL = 1e-9


def _close(a, b):
    return abs(a - b) < TOL


def wheel_velocity_unit_test_1():
    phi_l, phi_r = get_wheel_velocities(0.2, 0.0)
    expected = 0.2 / WHEEL_RADIUS_M
    return (_close(phi_l, expected) and _close(phi_r, expected)
            and phi_l > 0 and phi_r > 0)


def wheel_velocity_unit_test_2():
    phi_l, phi_r = get_wheel_velocities(-0.2, 0.0)
    expected = -0.2 / WHEEL_RADIUS_M
    return (_close(phi_l, expected) and _close(phi_r, expected)
            and phi_l < 0 and phi_r < 0)


def wheel_velocity_unit_test_3():
    phi_l, phi_r = get_wheel_velocities(0.0, 0.7)
    expected = 0.7 * (WHEELBASE_M / 2.0) / WHEEL_RADIUS_M
    return (_close(phi_l, -expected) and _close(phi_r, expected)
            and phi_l < 0 and phi_r > 0)


def wheel_velocity_unit_test_4():
    phi_l, phi_r = get_wheel_velocities(0.0, -0.7)
    expected = 0.7 * (WHEELBASE_M / 2.0) / WHEEL_RADIUS_M
    return (_close(phi_l, expected) and _close(phi_r, -expected)
            and phi_l > 0 and phi_r < 0)


def wheel_velocity_unit_test_5():
    phi_l, phi_r = get_wheel_velocities(0.0, 0.0)
    return _close(phi_l, 0.0) and _close(phi_r, 0.0)


# STUDENT TODO: run the 5 tests and print how many pass out of 5.
def run_wheel_velocity_unit_tests():
    tests = [wheel_velocity_unit_test_1, wheel_velocity_unit_test_2,
             wheel_velocity_unit_test_3, wheel_velocity_unit_test_4,
             wheel_velocity_unit_test_5]
    passed = 0
    for index, test in enumerate(tests, start=1):
        result = test()
        print(f"  wheel_velocity_unit_test_{index}: {'PASS' if result else 'FAIL'}")
        if result:
            passed += 1
    print(f"Wheel velocity tests: {passed}/5 passed")
    return passed


# =====================================================================
# PROVIDED - ROBOT SIMULATOR (do not change)
# =====================================================================
DT = 0.1                    # s, time step (same as the GUI control loop)
POS_TOL = 0.03              # m, goal reached if closer than this
ANG_TOL = 0.1               # rad, and heading error smaller than this


def wrap_angle(a):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def get_robot_velocities(phi_dot_l, phi_dot_r):
    """Wheel speeds (rad/s) -> robot (v, w)."""
    L = WHEELBASE_M / 2.0
    v = WHEEL_RADIUS_M * (phi_dot_r + phi_dot_l) / 2.0
    w = WHEEL_RADIUS_M * (phi_dot_r - phi_dot_l) / (2.0 * L)
    return v, w


def robot_simulator(X_tm1, phi_dot_l, phi_dot_r, delta_T, n_substeps=10):
    """Moves the robot for delta_T seconds with constant wheel speeds.
    Returns X_t (new pose [x, y, theta]) and the list of poses visited."""
    x, y, th = X_tm1
    h = delta_T / n_substeps
    poses = []
    for _ in range(n_substeps):
        v, w = get_robot_velocities(phi_dot_l, phi_dot_r)
        x += v * h * math.cos(th + w * h / 2.0)
        y += v * h * math.sin(th + w * h / 2.0)
        th = wrap_angle(th + w * h)
        poses.append([x, y, th])
    return [x, y, th], poses


def is_at_goal(X_des, X):
    """True if X is within POS_TOL and ANG_TOL of X_des."""
    return (math.hypot(X_des[0] - X[0], X_des[1] - X[1]) < POS_TOL and
            abs(wrap_angle(X_des[2] - X[2])) < ANG_TOL)


# =====================================================================
# PART 3 - POINT TRACKING
# =====================================================================
# STUDENT TODO: point_tracking_controller(X_des, X) -> (phi_dot_l, phi_dot_r)
#   X_des = desired pose [x, y, theta], X = current pose [x, y, theta].
#   Use the controller of Lecture 05A (rho, alpha, beta, gains, backwards
#   method), then get_wheel_velocities(v, w). Add any helper functions you need.
# Gains satisfy Lecture 05A slide 38:
# k_rho > 0, k_beta < 0, and k_alpha - k_rho > 0.
K_RHO = 0.35
K_ALPHA = 2.5
K_BETA = -0.7
K_THETA = 2.0     # Heading-only proportional gain (Lecture 05A slide 8).
FINAL_HEADING_DISTANCE_M = 0.015
# Velocity saturation (added after the robot experiments): with v = k_rho*rho
# alone, any goal farther than ~0.6 m commands more than the motors' ~9 rad/s
# and the robot arrives at full speed and overshoots. Capping v keeps the
# approach at V_MAX until rho < V_MAX/k_rho (0.57 m), then the proportional
# law slows the robot down smoothly. Both limits are well inside the wheel
# range (0.2 m/s = 6 rad/s; 1.5 rad/s = +/-4 rad/s wheel difference).
V_MAX = 0.20      # m/s
W_MAX = 1.5       # rad/s


def point_tracking_controller(X_des, X):
    x_des, y_des, theta_des = X_des
    x, y, theta = X
    dx = x_des - x
    dy = y_des - y

    # Lecture 05A slide 29: rho = sqrt(dx^2 + dy^2).
    rho = math.hypot(dx, dy)

    if rho < FINAL_HEADING_DISTANCE_M:
        # At the target the bearing atan2(dy, dx) is undefined.
        # Use the lecture's proportional heading control (slide 8).
        v = 0.0
        w = K_THETA * wrap_angle(theta_des - theta)
    else:
        # Slide 29: alpha = -theta + atan2(dy, dx).
        alpha = wrap_angle(-theta + math.atan2(dy, dx))

        if abs(alpha) > math.pi / 2.0:
            # Slides 42-43: drive backward rather than make a large turn.
            alpha = wrap_angle(-theta + math.atan2(-dy, -dx))
            v = -K_RHO * rho
        else:
            v = K_RHO * rho

        # Beta = -theta - alpha on slide 29 for theta_des = 0;
        # for an arbitrary desired heading, include theta_des.
        beta = wrap_angle(theta_des - theta - alpha)
        # Slides 36/43: proportional polar control law.
        w = K_ALPHA * alpha + K_BETA * beta

    # Saturate so the real robot never runs at full PWM into the goal.
    v = max(-V_MAX, min(V_MAX, v))
    w = max(-W_MAX, min(W_MAX, w))
    return get_wheel_velocities(v, w)


# STUDENT TODO: 6 unit tests (cases in the assignment), each returns
# (passed, visited_poses). Loop: point_tracking_controller -> robot_simulator
# until is_at_goal(X_des, X). Add a time limit so a test can fail.
TEST_TIME_LIMIT_S = 60.0


def _simulate_point_tracking(X_start, X_des):
    X = list(X_start)
    visited_poses = [list(X)]
    step = 0
    max_steps = int(TEST_TIME_LIMIT_S / DT)
    # Required while loop: control -> simulate -> append all visited poses.
    while not is_at_goal(X_des, X) and step < max_steps:
        phi_dot_l, phi_dot_r = point_tracking_controller(X_des, X)
        X, poses = robot_simulator(X, phi_dot_l, phi_dot_r, DT)
        visited_poses.extend(poses)
        step += 1
    return is_at_goal(X_des, X), visited_poses


def point_tracker_unit_test_1():
    return _simulate_point_tracking([0.0, 0.0, 0.0], [1.5, 1.5, 0.0])


def point_tracker_unit_test_2():
    return _simulate_point_tracking([0.0, 0.0, 0.0], [1.5, -1.5, 0.0])


def point_tracker_unit_test_3():
    return _simulate_point_tracking([0.0, 0.0, 0.0], [-1.5, 1.5, 0.0])


def point_tracker_unit_test_4():
    return _simulate_point_tracking([0.0, 0.0, 0.0], [-1.5, -1.5, 0.0])


def point_tracker_unit_test_5():
    return _simulate_point_tracking([0.0, 0.0, 0.0], [1.5, 1.5, math.pi])


def point_tracker_unit_test_6():
    return _simulate_point_tracking([1.0, 1.0, 0.0], [2.5, 2.5, math.pi])


# STUDENT TODO: run the 6 tests, print how many pass, plot all paths on one figure.
def run_point_tracker_unit_tests(plot=True):
    tests = [point_tracker_unit_test_1, point_tracker_unit_test_2,
             point_tracker_unit_test_3, point_tracker_unit_test_4,
             point_tracker_unit_test_5, point_tracker_unit_test_6]
    goals = [[1.5, 1.5, 0.0], [1.5, -1.5, 0.0],
             [-1.5, 1.5, 0.0], [-1.5, -1.5, 0.0],
             [1.5, 1.5, math.pi], [2.5, 2.5, math.pi]]
    passed_count = 0
    all_paths = []
    for i, test in enumerate(tests, start=1):
        passed, visited_poses = test()
        all_paths.append(visited_poses)
        if passed:
            passed_count += 1
        print(f"  point_tracker_unit_test_{i}: {'PASS' if passed else 'FAIL'} "
              f"({(len(visited_poses) - 1) * DT / 10.0:.1f} s)")
    print(f"Point tracking tests: {passed_count}/6 passed")

    if plot:
        # Matplotlib is used only because the assignment requires plots.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 7))
        for i, visited_poses in enumerate(all_paths):
            line, = plt.plot([p[0] for p in visited_poses],
                             [p[1] for p in visited_poses],
                             label=f"Point test {i + 1}")
            plt.plot(goals[i][0], goals[i][1], marker="x", linestyle="None",
                     markersize=8, color=line.get_color())
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Point-tracking unit tests: all six trajectories")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("point_tracking_tests.png", dpi=170)
        plt.close()
        print("Saved point_tracking_tests.png")
    return passed_count


# Extra point-tracking cases required by Step 4 of the assignment.
def extra_point_tracker_unit_test_1():
    # Goal behind the robot: backwards method should be used.
    return _simulate_point_tracking([0.0, 0.0, 0.0], [-2.0, 0.0, 0.0])


def extra_point_tracker_unit_test_2():
    # In-place orientation correction at the destination.
    return _simulate_point_tracking([0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0])


def extra_point_tracker_unit_test_3():
    # Heading crosses the -pi/pi boundary by a small amount.
    return _simulate_point_tracking([0.0, 0.0, math.pi - 0.2],
                                    [0.0, 0.0, -math.pi + 0.2])


def extra_point_tracker_unit_test_4():
    # Axis-aligned straight-line movement.
    return _simulate_point_tracking([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])


def run_extra_point_tests():
    tests = [extra_point_tracker_unit_test_1, extra_point_tracker_unit_test_2,
             extra_point_tracker_unit_test_3, extra_point_tracker_unit_test_4]
    passed_count = 0
    for i, test in enumerate(tests, start=1):
        passed, _ = test()
        if passed:
            passed_count += 1
        print(f"  extra_point_tracker_unit_test_{i}: {'PASS' if passed else 'FAIL'}")
    print(f"Extra point tests: {passed_count}/{len(tests)} passed")
    return passed_count


# =====================================================================
# PART 4 - PATH TRACKING
# =====================================================================
# STUDENT TODO: path_tracking_controller(P, X) -> (phi_dot_l, phi_dot_r)
#   P = path, list of poses [[x, y, theta], ...], X = current pose.
#   Use the method of Lecture 06A (closest point, look-ahead distance Delta,
#   point tracking to that point). Add any helper functions you need.
LOOKAHEAD_DELTA_M = 0.25


def _path_lookahead_pose(P, X, delta):
    """Closest projection on the polyline, then delta meters farther along it."""
    closest_dist_squared = float("inf")
    closest_segment = 0
    closest_t = 0.0

    # Find p_c,t, the point of P closest to the robot's (x,y).
    for i in range(len(P) - 1):
        x0, y0 = P[i][0], P[i][1]
        dx, dy = P[i + 1][0] - x0, P[i + 1][1] - y0
        segment_length_squared = dx * dx + dy * dy
        if segment_length_squared == 0.0:
            t = 0.0  # In-place change of desired heading (zero-length segment).
        else:
            t = ((X[0] - x0) * dx + (X[1] - y0) * dy) / segment_length_squared
            t = max(0.0, min(1.0, t))
        closest_x, closest_y = x0 + t * dx, y0 + t * dy
        d2 = (X[0] - closest_x) ** 2 + (X[1] - closest_y) ** 2
        if d2 < closest_dist_squared:
            closest_dist_squared = d2
            closest_segment = i
            closest_t = t

    # Move delta meters forward *along the path*, not straight from the robot.
    remaining_delta = delta
    for i in range(closest_segment, len(P) - 1):
        a, b = P[i], P[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        segment_length = math.hypot(dx, dy)
        if segment_length == 0.0:
            continue
        start_t = closest_t if i == closest_segment else 0.0
        segment_remaining = (1.0 - start_t) * segment_length
        if remaining_delta <= segment_remaining:
            target_t = start_t + remaining_delta / segment_length
            target_x = a[0] + target_t * dx
            target_y = a[1] + target_t * dy
            # Interpolate the segment heading by its shortest angular change.
            target_theta = wrap_angle(a[2] + target_t * wrap_angle(b[2] - a[2]))
            return [target_x, target_y, target_theta]
        remaining_delta -= segment_remaining

    # If delta extends past the end of P, track its final pose exactly.
    return list(P[-1])


def path_tracking_controller(P, X):
    if not P:
        return 0.0, 0.0
    if len(P) == 1:
        return point_tracking_controller(P[0], X)
    X_des = _path_lookahead_pose(P, X, LOOKAHEAD_DELTA_M)
    return point_tracking_controller(X_des, X)


# STUDENT TODO: 3 unit tests (cases in the assignment), each returns
# (passed, visited_poses). Stop when is_at_goal(P[-1], X).
PATH_POS_TOL = 0.02  # Assignment's stricter path-position tolerance.


def _simulate_path_tracking(P, X_start):
    X = list(X_start)
    visited_poses = [list(X)]
    step = 0
    max_steps = int(TEST_TIME_LIMIT_S / DT)
    # Continue until both provided is_at_goal and the assignment's
    # 0.02 m position tolerance hold. This tests heading to 0.1 rad,
    # stricter than the assignment's 0.3 rad criterion.
    while (not is_at_goal(P[-1], X) or
           math.hypot(P[-1][0] - X[0], P[-1][1] - X[1]) >= PATH_POS_TOL):
        if step >= max_steps:
            break
        phi_dot_l, phi_dot_r = path_tracking_controller(P, X)
        X, poses = robot_simulator(X, phi_dot_l, phi_dot_r, DT)
        visited_poses.extend(poses)
        step += 1
    passed = (is_at_goal(P[-1], X) and
              math.hypot(P[-1][0] - X[0], P[-1][1] - X[1]) < PATH_POS_TOL)
    return passed, visited_poses


def path_tracker_unit_test_1():
    P = [[0.0, 0.1, 0.0], [2.0, 0.1, 0.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


def path_tracker_unit_test_2():
    P = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [1.0, 0.0, math.pi / 2.0], [1.0, 1.0, math.pi / 2.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


def path_tracker_unit_test_3():
    P = [[0.0, 0.0, math.pi / 4.0], [2.0, 2.0, math.pi / 4.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


# STUDENT TODO: run the 3 tests, print how many pass, plot each path as a subfigure.
def run_path_tracker_unit_tests(plot=True):
    tests = [path_tracker_unit_test_1, path_tracker_unit_test_2,
             path_tracker_unit_test_3]
    paths = [
        [[0.0, 0.1, 0.0], [2.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [1.0, 0.0, math.pi / 2.0], [1.0, 1.0, math.pi / 2.0]],
        [[0.0, 0.0, math.pi / 4.0], [2.0, 2.0, math.pi / 4.0]]
    ]
    all_visited = []
    passed_count = 0
    for i, test in enumerate(tests, start=1):
        passed, visited_poses = test()
        all_visited.append(visited_poses)
        if passed:
            passed_count += 1
        print(f"  path_tracker_unit_test_{i}: {'PASS' if passed else 'FAIL'} "
              f"({(len(visited_poses) - 1) * DT / 10.0:.1f} s)")
    print(f"Path tracking tests: {passed_count}/3 passed")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axes):
            P = paths[i]
            visited = all_visited[i]
            ax.plot([p[0] for p in P], [p[1] for p in P], "--",
                    label="Desired path")
            ax.plot([p[0] for p in visited], [p[1] for p in visited],
                    label="Simulated path")
            ax.plot(P[-1][0], P[-1][1], marker="x", markersize=8)
            ax.set_title(f"Path test {i + 1}" + (" (y enlarged)" if i == 0 else ""))
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            # Test 1 has a 2 m x-range but only 0.1 m of y variation;
            # enlarge y so the small cross-track correction can be seen.
            ax.set_aspect("auto" if i == 0 else "equal", adjustable="box")
            ax.grid(True)
            ax.legend()
        fig.suptitle("Path-tracking tests: reference vs. followed path")
        fig.tight_layout()
        fig.savefig("path_tracking_tests.png", dpi=170)
        plt.close(fig)
        print("Saved path_tracking_tests.png")
    return passed_count


# Extra path-tracking cases required by Step 4 (run separately).
def extra_path_tracker_unit_test_1():
    P = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


def extra_path_tracker_unit_test_2():
    P = [[0.0, 0.2, 0.0], [1.5, 0.2, 0.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


def extra_path_tracker_unit_test_3():
    P = [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
         [0.8, 0.8, math.pi / 2.0]]
    return _simulate_path_tracking(P, [0.0, 0.0, 0.0])


def run_extra_path_tests():
    tests = [extra_path_tracker_unit_test_1, extra_path_tracker_unit_test_2,
             extra_path_tracker_unit_test_3]
    passed_count = 0
    for i, test in enumerate(tests, start=1):
        passed, _ = test()
        if passed:
            passed_count += 1
        print(f"  extra_path_tracker_unit_test_{i}: {'PASS' if passed else 'FAIL'}")
    print(f"Extra path tests: {passed_count}/{len(tests)} passed")
    return passed_count


if __name__ == "__main__":
    print("=== Part 1: wheel velocities ===")
    run_wheel_velocity_unit_tests()
    print("\n=== Part 3: point tracking ===")
    run_point_tracker_unit_tests()
    try:
        run_extra_point_tests()
    except NameError:
        pass
    print("\n=== Part 4: path tracking ===")
    run_path_tracker_unit_tests()
