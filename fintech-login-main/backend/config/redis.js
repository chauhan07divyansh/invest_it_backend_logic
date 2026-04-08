const Redis = require('ioredis');
const logger = require('../utils/logger');

let client;

const getRedisClient = () => {
  if (client) return client;

  const options = process.env.REDIS_URL
    ? process.env.REDIS_URL
    : {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      password: process.env.REDIS_PASSWORD || undefined,
      retryStrategy: (times) => {
        const delay = Math.min(times * 100, 3000);
        logger.warn(`Redis reconnect attempt #${times}, retrying in ${delay}ms`);
        return delay;
      },
      maxRetriesPerRequest: 3,
      enableOfflineQueue: true,
      lazyConnect: false,
    };

  client = new Redis(options);

  client.on('connect', () => logger.info('Redis client connected'));
  client.on('ready', () => logger.info('Redis client ready'));
  client.on('error', (err) => logger.error('Redis client error', { error: err.message }));
  client.on('close', () => logger.warn('Redis connection closed'));
  client.on('reconnecting', () => logger.warn('Redis reconnecting...'));

  return client;
};

// Helper wrappers
const redisSet = async (key, value, ttlSeconds) => {
  const redis = getRedisClient();
  const serialized = typeof value === 'object' ? JSON.stringify(value) : String(value);
  if (ttlSeconds) {
    return redis.setex(key, ttlSeconds, serialized);
  }
  return redis.set(key, serialized);
};

const redisGet = async (key) => {
  const redis = getRedisClient();
  const value = await redis.get(key);
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const redisDel = async (key) => {
  const redis = getRedisClient();
  return redis.del(key);
};

const redisExists = async (key) => {
  const redis = getRedisClient();
  return redis.exists(key);
};

module.exports = { getRedisClient, redisSet, redisGet, redisDel, redisExists };