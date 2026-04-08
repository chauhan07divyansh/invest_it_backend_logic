const { verifyAccessToken, isTokenBlacklisted } = require('../utils/jwtUtils');
const logger = require('../utils/logger');

/**
 * Protects routes by verifying the Bearer access token.
 * Also checks the Redis blacklist (handles logout-before-expiry).
 */
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        success: false,
        message: 'Access token required',
        code: 'TOKEN_MISSING',
      });
    }

    const token = authHeader.split(' ')[1];

    // Check blacklist first (fast Redis hit)
    const blacklisted = await isTokenBlacklisted(token);
    if (blacklisted) {
      return res.status(401).json({
        success: false,
        message: 'Token has been revoked',
        code: 'TOKEN_REVOKED',
      });
    }

    const decoded = verifyAccessToken(token);
    req.user = decoded;
    req.token = token;
    next();
  } catch (err) {
    logger.warn('Authentication failed', { error: err.message, ip: req.ip });

    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({
        success: false,
        message: 'Access token expired',
        code: 'TOKEN_EXPIRED',
      });
    }

    return res.status(401).json({
      success: false,
      message: 'Invalid access token',
      code: 'TOKEN_INVALID',
    });
  }
};

module.exports = { authenticate };