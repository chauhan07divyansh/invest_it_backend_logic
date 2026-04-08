const express = require('express');
const router = express.Router();

const { register, login, refresh, logout, me } = require('../controllers/authController');
const { authenticate } = require('../middleware/auth');
const { registerValidation, loginValidation, refreshTokenValidation } = require('../middleware/validators');
const { createAuthLimiter } = require('../middleware/rateLimiter');

const authLimiter = createAuthLimiter();

// Public routes
router.post('/register', authLimiter, registerValidation, register);
router.post('/login',    authLimiter, loginValidation,    login);
router.post('/refresh',  refreshTokenValidation,          refresh);

// Protected routes
router.post('/logout', authenticate, logout);
router.get('/me',      authenticate, me);

module.exports = router;