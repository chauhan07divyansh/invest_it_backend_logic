"""
Sentiquant — Retention Check (run from Render shell)
Usage:
    python3 -c "$(cat retention_check.py)"
Or paste the whole block into:
    python3 - <<'EOF'
    ...
    EOF
Reuses your existing Flask app + db session — no new connection needed.
"""
import sys
sys.path.insert(0, '/app')
from main import app, db
from sqlalchemy import text

with app.app_context():

    print("="*70)
    print("0. SANITY CHECK — do you have enough data yet?")
    print("="*70)
    row = db.session.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM users) AS total_users,
            (SELECT COUNT(*) FROM users WHERE is_verified = true) AS verified_users,
            (SELECT COUNT(DISTINCT user_id) FROM user_usage WHERE user_id IS NOT NULL) AS users_with_any_activity,
            (SELECT MIN(created_at)::date FROM users) AS first_signup,
            (SELECT MAX(created_at)::date FROM users) AS most_recent_signup,
            (SELECT COUNT(*) FROM user_usage) AS total_actions_logged
    """)).fetchone()
    print(f"  Total users:            {row.total_users}")
    print(f"  Verified users:         {row.verified_users}")
    print(f"  Users with any activity:{row.users_with_any_activity}")
    print(f"  First signup:           {row.first_signup}")
    print(f"  Most recent signup:     {row.most_recent_signup}")
    print(f"  Total actions logged:   {row.total_actions_logged}")
    print()

    print("="*70)
    print("1. D1 / D7 / D30 RETENTION (% who came back and took an action)")
    print("="*70)
    rows = db.session.execute(text("""
        WITH signups AS (
            SELECT id AS user_id, created_at::date AS signup_date
            FROM users
        ),
        actions AS (
            SELECT DISTINCT user_id, timestamp::date AS action_date
            FROM user_usage
            WHERE user_id IS NOT NULL
        ),
        cohort AS (
            SELECT
                s.user_id,
                s.signup_date,
                MAX(CASE WHEN a.action_date BETWEEN s.signup_date + 1 AND s.signup_date + 1 THEN 1 ELSE 0 END) AS d1,
                MAX(CASE WHEN a.action_date BETWEEN s.signup_date + 1 AND s.signup_date + 7 THEN 1 ELSE 0 END) AS d7,
                MAX(CASE WHEN a.action_date BETWEEN s.signup_date + 1 AND s.signup_date + 30 THEN 1 ELSE 0 END) AS d30
            FROM signups s
            LEFT JOIN actions a ON a.user_id = s.user_id
            GROUP BY s.user_id, s.signup_date
        )
        SELECT
            COUNT(*) AS total_signups,
            ROUND(100.0 * SUM(d1)  / COUNT(*), 1) AS d1_retention_pct,
            ROUND(100.0 * SUM(d7)  / COUNT(*), 1) AS d7_retention_pct,
            ROUND(100.0 * SUM(d30) / COUNT(*), 1) AS d30_retention_pct
        FROM cohort
        WHERE signup_date <= CURRENT_DATE - 30
    """)).fetchone()
    if rows and rows.total_signups:
        print(f"  Signups eligible (30+ days old): {rows.total_signups}")
        print(f"  D1 retention:  {rows.d1_retention_pct}%")
        print(f"  D7 retention:  {rows.d7_retention_pct}%")
        print(f"  D30 retention: {rows.d30_retention_pct}%")
    else:
        print("  No signups yet that are 30+ days old — too early to measure D30.")
    print()

    print("="*70)
    print("2. ACTIVITY DISTRIBUTION (how many days did each user ever use it?)")
    print("="*70)
    rows = db.session.execute(text("""
        SELECT
            activity_count,
            COUNT(*) AS num_users
        FROM (
            SELECT user_id, COUNT(DISTINCT timestamp::date) AS activity_count
            FROM user_usage
            WHERE user_id IS NOT NULL
            GROUP BY user_id
        ) t
        GROUP BY activity_count
        ORDER BY activity_count
    """)).fetchall()
    if rows:
        for r in rows:
            print(f"  Used on {r.activity_count} distinct day(s): {r.num_users} user(s)")
    else:
        print("  No user_usage rows yet.")
    print()

    print("="*70)
    print("3. LOGIN RETENTION vs PRODUCT RETENTION (the gap that matters)")
    print("="*70)
    row = db.session.execute(text("""
        WITH logins AS (
            SELECT DISTINCT user_id, created_at::date AS login_date
            FROM login_audit
            WHERE status = 'success' AND user_id IS NOT NULL
        ),
        actions AS (
            SELECT DISTINCT user_id, timestamp::date AS action_date
            FROM user_usage
            WHERE user_id IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT l.user_id) AS users_who_logged_in_again,
            COUNT(DISTINCT a.user_id) AS users_who_took_action,
            ROUND(100.0 * COUNT(DISTINCT a.user_id) / NULLIF(COUNT(DISTINCT l.user_id), 0), 1) AS pct_logins_that_convert_to_action
        FROM logins l
        LEFT JOIN actions a ON a.user_id = l.user_id AND a.action_date = l.login_date
    """)).fetchone()
    print(f"  Users who logged in (ever, post-signup): {row.users_who_logged_in_again}")
    print(f"  Users who took an in-session action:     {row.users_who_took_action}")
    print(f"  % of login-days that converted to action: {row.pct_logins_that_convert_to_action}%")
    print()

    print("="*70)
    print("4. TOP 20 BY ACTIVITY (your power users)")
    print("="*70)
    rows = db.session.execute(text("""
        SELECT
            u.email,
            u.plan,
            u.created_at::date AS signup_date,
            COUNT(DISTINCT uu.timestamp::date) AS active_days,
            COUNT(uu.id) AS total_actions,
            MAX(uu.timestamp)::date AS last_active
        FROM users u
        LEFT JOIN user_usage uu ON uu.user_id = u.id
        GROUP BY u.id, u.email, u.plan, u.created_at
        ORDER BY active_days DESC, total_actions DESC
        LIMIT 20
    """)).fetchall()
    if rows:
        print(f"  {'Email':<35}{'Plan':<8}{'Signup':<12}{'Active days':<12}{'Actions':<10}{'Last active'}")
        for r in rows:
            print(f"  {r.email[:34]:<35}{r.plan:<8}{str(r.signup_date):<12}{r.active_days:<12}{r.total_actions:<10}{r.last_active}")
    else:
        print("  No users found.")
    print()

    print("="*70)
    print("DONE — see retention_queries.sql.md for the cohort-week chart query")
    print("="*70)
