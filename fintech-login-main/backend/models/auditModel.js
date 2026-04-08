const { query } = require('../config/database');

const AuditModel = {
  /**
   * Records a login attempt (success or failure) to the audit log.
   */
  log: async ({ userId, email, ipAddress, userAgent, status, failureReason }) => {
    await query(
      `INSERT INTO login_audit (user_id, email, ip_address, user_agent, status, failure_reason)
       VALUES ($1, $2, $3::INET, $4, $5, $6)`,
      [userId || null, email, ipAddress, userAgent, status, failureReason || null]
    );
  },

  /**
   * Returns the last N login events for a user (for the user's own audit trail).
   */
  getRecentForUser: async (userId, limit = 10) => {
    const result = await query(
      `SELECT ip_address, user_agent, status, failure_reason, created_at
       FROM login_audit
       WHERE user_id = $1
       ORDER BY created_at DESC
       LIMIT $2`,
      [userId, limit]
    );
    return result.rows;
  },
};

module.exports = AuditModel;