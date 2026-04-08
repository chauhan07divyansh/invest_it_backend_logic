const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const { redisSet, redisGet, redisDel, redisExists } = require('../config/redis');
const logger = require('./logger');

const ACCESS_TOKEN_TTL = 15 * 60;        // 15 minutes in seconds
const REFRESH_TOKEN_TTL = 7 * 24 * 3600; // 7 days in seconds

/**
 * Generates a signed access token (short-lived).
 */
const generateAccessToken = (payload) => {
  return jwt.sign(payload, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN || '15m',
    issuer: 'fintech-auth',
    audience: 'fintech-app',
  });
};

/**
 * Generates a signed refresh token (long-lived) and stores it in Redis.
 */
const generateRefreshToken = async (userId) => {
  const tokenId = uuidv4();
  const token = jwt.sign(
    { userId, tokenId },
    process.env.JWT_REFRESH_SECRET,
    {
      expiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
      issuer: 'fintech-auth',
      audience: 'fintech-app',
    }
  );

  // Store in Redis: key = refresh:{userId}:{tokenId}
  await redisSet(`refresh:${userId}:${tokenId}`, { userId, tokenId }, REFRESH_TOKEN_TTL);

  return { token, tokenId };
};

/**
 * Verifies an access token. Returns decoded payload or throws.
 */
const verifyAccessToken = (token) => {
  return jwt.verify(token, process.env.JWT_SECRET, {
    issuer: 'fintech-auth',
    audience: 'fintech-app',
  });
};

/**
 * Verifies a refresh token and checks Redis for validity.
 */
const verifyRefreshToken = async (token) => {
  const decoded = jwt.verify(token, process.env.JWT_REFRESH_SECRET, {
    issuer: 'fintech-auth',
    audience: 'fintech-app',
  });

  const key = `refresh:${decoded.userId}:${decoded.tokenId}`;
  const stored = await redisGet(key);

  if (!stored) {
    throw new Error('Refresh token not found or expired');
  }

  return decoded;
};

/**
 * Revokes a specific refresh token (logout).
 */
const revokeRefreshToken = async (userId, tokenId) => {
  await redisDel(`refresh:${userId}:${tokenId}`);
};

/**
 * Revokes ALL refresh tokens for a user (force logout everywhere).
 */
const revokeAllUserTokens = async (userId) => {
  const redis = require('../config/redis').getRedisClient();
  const keys = await redis.keys(`refresh:${userId}:*`);
  if (keys.length > 0) {
    await redis.del(...keys);
    logger.info(`Revoked ${keys.length} tokens for user ${userId}`);
  }
};

/**
 * Blacklists an access token (for logout before expiry).
 */
const blacklistAccessToken = async (token) => {
  try {
    const decoded = verifyAccessToken(token);
    const ttl = decoded.exp - Math.floor(Date.now() / 1000);
    if (ttl > 0) {
      await redisSet(`blacklist:${token}`, '1', ttl);
    }
  } catch {
    // Token already invalid, no need to blacklist
  }
};

/**
 * Checks if an access token is blacklisted.
 */
const isTokenBlacklisted = async (token) => {
  const exists = await redisExists(`blacklist:${token}`);
  return exists === 1;
};

module.exports = {
  generateAccessToken,
  generateRefreshToken,
  verifyAccessToken,
  verifyRefreshToken,
  revokeRefreshToken,
  revokeAllUserTokens,
  blacklistAccessToken,
  isTokenBlacklisted,
};