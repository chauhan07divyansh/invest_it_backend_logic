const UserModel = require('../models/userModel');
const AuditModel = require('../models/auditModel');
const {
  generateAccessToken,
  generateRefreshToken,
  verifyRefreshToken,
  revokeRefreshToken,
  blacklistAccessToken,
} = require('../utils/jwtUtils');
const { sendLoginAlert, sendWelcomeEmail } = require('../utils/emailService');
const logger = require('../utils/logger');

/**
 * Extracts the real IP, accounting for proxies.
 */
const getClientIp = (req) => {
  return (
    req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
    req.headers['x-real-ip'] ||
    req.connection?.remoteAddress ||
    req.ip
  );
};

// ─────────────────────────────────────────────
//  POST /api/auth/register
// ─────────────────────────────────────────────
const register = async (req, res) => {
  const { email, password, firstName, lastName } = req.body;

  try {
    // Check for existing user
    const existing = await UserModel.findByEmail(email);
    if (existing) {
      return res.status(409).json({
        success: false,
        message: 'An account with this email already exists',
        code: 'EMAIL_TAKEN',
      });
    }

    const user = await UserModel.create({ email, password, firstName, lastName });

    // Send welcome email (non-blocking)
    sendWelcomeEmail({ userEmail: user.email, firstName: user.first_name });

    logger.info('User registered', { userId: user.id, email: user.email });

    return res.status(201).json({
      success: true,
      message: 'Account created successfully',
      user: {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
      },
    });
  } catch (err) {
    logger.error('Registration error', { error: err.message, email });
    return res.status(500).json({
      success: false,
      message: 'Registration failed. Please try again.',
      code: 'INTERNAL_ERROR',
    });
  }
};

// ─────────────────────────────────────────────
//  POST /api/auth/login
// ─────────────────────────────────────────────
const login = async (req, res) => {
  const { email, password } = req.body;
  const ip = getClientIp(req);
  const userAgent = req.headers['user-agent'] || 'Unknown';

  try {
    const user = await UserModel.findByEmail(email);

    // User not found — same error as wrong password (prevent email enumeration)
    if (!user) {
      await AuditModel.log({
        email,
        ipAddress: ip,
        userAgent,
        status: 'failed',
        failureReason: 'user_not_found',
      });
      return res.status(401).json({
        success: false,
        message: 'Invalid email or password',
        code: 'INVALID_CREDENTIALS',
      });
    }

    // Account locked?
    if (UserModel.isAccountLocked(user)) {
      const lockedUntil = new Date(user.locked_until);
      await AuditModel.log({
        userId: user.id,
        email,
        ipAddress: ip,
        userAgent,
        status: 'locked',
        failureReason: 'account_locked',
      });
      return res.status(423).json({
        success: false,
        message: `Account temporarily locked. Try again after ${lockedUntil.toISOString()}`,
        code: 'ACCOUNT_LOCKED',
        lockedUntil: lockedUntil.toISOString(),
      });
    }

    // Account inactive?
    if (!user.is_active) {
      return res.status(403).json({
        success: false,
        message: 'Account is disabled. Please contact support.',
        code: 'ACCOUNT_DISABLED',
      });
    }

    // Verify password
    const passwordValid = await UserModel.verifyPassword(password, user.password_hash);
    if (!passwordValid) {
      const { failed_login_attempts } = await UserModel.recordFailedAttempt(user.id);
      const remaining = Math.max(0, 5 - failed_login_attempts);

      await AuditModel.log({
        userId: user.id,
        email,
        ipAddress: ip,
        userAgent,
        status: 'failed',
        failureReason: 'wrong_password',
      });

      logger.warn('Failed login attempt', { userId: user.id, ip, attempts: failed_login_attempts });

      return res.status(401).json({
        success: false,
        message: remaining > 0
          ? `Invalid email or password. ${remaining} attempt${remaining !== 1 ? 's' : ''} remaining.`
          : 'Account locked due to too many failed attempts.',
        code: 'INVALID_CREDENTIALS',
        attemptsRemaining: remaining,
      });
    }

    // ✅ Login success — reset failed attempts
    await UserModel.resetFailedAttempts(user.id);

    // Generate tokens
    const tokenPayload = {
      userId: user.id,
      email: user.email,
      firstName: user.first_name,
    };
    const accessToken = generateAccessToken(tokenPayload);
    const { token: refreshToken, tokenId } = await generateRefreshToken(user.id);

    // Audit log
    await AuditModel.log({
      userId: user.id,
      email,
      ipAddress: ip,
      userAgent,
      status: 'success',
    });

    // Send admin notification email (non-blocking)
    sendLoginAlert({
      userEmail: user.email,
      ip,
      userAgent,
      timestamp: new Date().toLocaleString('en-US', { timeZone: 'UTC' }) + ' UTC',
    });

    logger.info('Successful login', { userId: user.id, email: user.email, ip });

    return res.status(200).json({
      success: true,
      message: 'Login successful',
      accessToken,
      refreshToken,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
      },
    });
  } catch (err) {
    logger.error('Login error', { error: err.message, email, ip });
    return res.status(500).json({
      success: false,
      message: 'Login failed. Please try again.',
      code: 'INTERNAL_ERROR',
    });
  }
};

// ─────────────────────────────────────────────
//  POST /api/auth/refresh
// ─────────────────────────────────────────────
const refresh = async (req, res) => {
  const { refreshToken } = req.body;

  try {
    const decoded = await verifyRefreshToken(refreshToken);

    const user = await UserModel.findById(decoded.userId);
    if (!user || !user.is_active) {
      return res.status(401).json({
        success: false,
        message: 'User not found or inactive',
        code: 'USER_INVALID',
      });
    }

    // Rotate: revoke old, issue new pair
    await revokeRefreshToken(decoded.userId, decoded.tokenId);

    const tokenPayload = {
      userId: user.id,
      email: user.email,
      firstName: user.first_name,
    };
    const newAccessToken = generateAccessToken(tokenPayload);
    const { token: newRefreshToken } = await generateRefreshToken(user.id);

    return res.status(200).json({
      success: true,
      accessToken: newAccessToken,
      refreshToken: newRefreshToken,
    });
  } catch (err) {
    logger.warn('Token refresh failed', { error: err.message });
    return res.status(401).json({
      success: false,
      message: 'Invalid or expired refresh token',
      code: 'REFRESH_TOKEN_INVALID',
    });
  }
};

// ─────────────────────────────────────────────
//  POST /api/auth/logout
// ─────────────────────────────────────────────
const logout = async (req, res) => {
  const { refreshToken } = req.body;
  const accessToken = req.token;

  try {
    // Blacklist current access token
    if (accessToken) {
      await blacklistAccessToken(accessToken);
    }

    // Revoke refresh token if provided
    if (refreshToken) {
      try {
        const decoded = await verifyRefreshToken(refreshToken);
        await revokeRefreshToken(decoded.userId, decoded.tokenId);
      } catch {
        // Refresh token already invalid — that's fine
      }
    }

    logger.info('User logged out', { userId: req.user?.userId });

    return res.status(200).json({
      success: true,
      message: 'Logged out successfully',
    });
  } catch (err) {
    logger.error('Logout error', { error: err.message });
    return res.status(500).json({
      success: false,
      message: 'Logout failed',
      code: 'INTERNAL_ERROR',
    });
  }
};

// ─────────────────────────────────────────────
//  GET /api/auth/me
// ─────────────────────────────────────────────
const me = async (req, res) => {
  try {
    const user = await UserModel.findById(req.user.userId);
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    return res.status(200).json({ success: true, user });
  } catch (err) {
    logger.error('Me endpoint error', { error: err.message });
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

module.exports = { register, login, refresh, logout, me };