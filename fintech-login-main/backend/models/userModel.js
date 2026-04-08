const { query } = require('../config/database');
const bcrypt = require('bcryptjs');

const SALT_ROUNDS = 12;
const MAX_FAILED_ATTEMPTS = 5;
const LOCKOUT_DURATION_MINUTES = 15;

const UserModel = {
  /**
   * Find user by email. Returns null if not found.
   */
  findByEmail: async (email) => {
    const result = await query(
      'SELECT * FROM users WHERE email = $1',
      [email.toLowerCase().trim()]
    );
    return result.rows[0] || null;
  },

  /**
   * Find user by ID.
   */
  findById: async (id) => {
    const result = await query(
      'SELECT id, email, first_name, last_name, is_active, is_email_verified, created_at FROM users WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  },

  /**
   * Create a new user with a hashed password.
   */
  create: async ({ email, password, firstName, lastName }) => {
    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const result = await query(
      `INSERT INTO users (email, password_hash, first_name, last_name)
       VALUES ($1, $2, $3, $4)
       RETURNING id, email, first_name, last_name, is_active, created_at`,
      [email.toLowerCase().trim(), passwordHash, firstName, lastName]
    );
    return result.rows[0];
  },

  /**
   * Verifies a plaintext password against the stored hash.
   */
  verifyPassword: async (plaintext, hash) => {
    return bcrypt.compare(plaintext, hash);
  },

  /**
   * Checks if the user account is locked due to too many failed attempts.
   */
  isAccountLocked: (user) => {
    if (!user.locked_until) return false;
    return new Date(user.locked_until) > new Date();
  },

  /**
   * Records a failed login attempt and locks the account if threshold is reached.
   */
  recordFailedAttempt: async (userId) => {
    const result = await query(
      `UPDATE users
       SET failed_login_attempts = failed_login_attempts + 1,
           locked_until = CASE
             WHEN failed_login_attempts + 1 >= $1
             THEN NOW() + INTERVAL '${LOCKOUT_DURATION_MINUTES} minutes'
             ELSE locked_until
           END
       WHERE id = $2
       RETURNING failed_login_attempts, locked_until`,
      [MAX_FAILED_ATTEMPTS, userId]
    );
    return result.rows[0];
  },

  /**
   * Resets failed attempts and clears lockout after successful login.
   */
  resetFailedAttempts: async (userId) => {
    await query(
      `UPDATE users
       SET failed_login_attempts = 0, locked_until = NULL
       WHERE id = $1`,
      [userId]
    );
  },
};

module.exports = UserModel;