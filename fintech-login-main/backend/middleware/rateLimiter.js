const rateLimit = require('express-rate-limit');
const { RedisStore } = require('rate-limit-redis');
const { getRedisClient } = require('../config/redis');
const logger = require('../utils/logger');

/**
 * Strict rate limiter for auth endpoints.
 * 5 attempts per 15 minutes per IP, backed by Redis.
 */
const createAuthLimiter = () => {
  const redis = getRedisClient();

  return rateLimit({
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 5,
    standardHeaders: true,
    legacyHeaders: false,
    store: new RedisStore({
      sendCommand: (...args) => redis.call(...args),
      prefix: 'rl:auth:',
    }),
    keyGenerator: (req) => {
      // Rate limit by IP + email combo for smarter throttling
      const email = req.body?.email?.toLowerCase()?.trim() || 'unknown';
      return `${req.ip}:${email}`;
    },
    handler: (req, res) => {
      logger.warn('Rate limit exceeded', {
        ip: req.ip,
        email: req.body?.email,
        path: req.path,
      });
      res.status(429).json({
        success: false,
        message: 'Too many login attempts. Please try again in 15 minutes.',
        code: 'RATE_LIMIT_EXCEEDED',
        retryAfter: Math.ceil(
          (req.rateLimit.resetTime - Date.now()) / 1000
        ),
      });
    },
    skip: (req) => {
      // Skip rate limiting in test environment
      return process.env.NODE_ENV === 'test';
    },
  });
};

/**
 * General API rate limiter (less strict).
 */
const createGeneralLimiter = () => {
  const redis = getRedisClient();

  return rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
    store: new RedisStore({
      sendCommand: (...args) => redis.call(...args),
      prefix: 'rl:general:',
    }),
  });
};

module.exports = { createAuthLimiter, createGeneralLimiter };