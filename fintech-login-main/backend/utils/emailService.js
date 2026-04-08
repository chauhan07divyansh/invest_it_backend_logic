const nodemailer = require('nodemailer');
const logger = require('./logger');

let transporter;

const getTransporter = () => {
  if (transporter) return transporter;

  transporter = nodemailer.createTransport({
    host: 'smtp.zoho.in',
    port: 465,
    secure: true,
    auth: {
      user: process.env.ZOHO_USER,
      pass: process.env.ZOHO_PASSWORD,
    },
    pool: true,
    maxConnections: 5,
    maxMessages: 100,
  });

  return transporter;
};

/**
 * Sends a login alert to the admin inbox.
 * Called on every successful login.
 */
const sendLoginAlert = async ({ userEmail, ip, userAgent, timestamp, location }) => {
  const transport = getTransporter();

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login Alert</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f5f7; }
    .wrapper { max-width: 600px; margin: 40px auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.07); }
    .header { background: #0d1117; padding: 32px 40px; border-bottom: 3px solid #00d4aa; }
    .header h1 { color: #fff; font-size: 20px; font-weight: 600; letter-spacing: -0.3px; }
    .header p { color: #8b949e; font-size: 13px; margin-top: 4px; }
    .badge { display: inline-block; background: #00d4aa20; color: #00d4aa; border: 1px solid #00d4aa40; border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 600; margin-top: 8px; letter-spacing: 0.5px; text-transform: uppercase; }
    .body { padding: 32px 40px; }
    .alert-icon { width: 48px; height: 48px; background: #fff8e1; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; }
    .title { font-size: 22px; font-weight: 700; color: #0d1117; margin-bottom: 8px; }
    .subtitle { font-size: 14px; color: #656d76; margin-bottom: 28px; }
    .detail-grid { background: #f6f8fa; border-radius: 10px; padding: 20px; border: 1px solid #e9ecef; }
    .detail-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #e9ecef; }
    .detail-row:last-child { border-bottom: none; }
    .detail-label { font-size: 12px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .detail-value { font-size: 14px; color: #0d1117; font-weight: 500; text-align: right; max-width: 300px; word-break: break-all; }
    .footer { padding: 20px 40px; background: #f6f8fa; border-top: 1px solid #e9ecef; text-align: center; }
    .footer p { font-size: 12px; color: #8b949e; }
    .footer a { color: #00d4aa; text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>⚡ FintechApp</h1>
      <p>Security & Authentication System</p>
      <span class="badge">Login Detected</span>
    </div>
    <div class="body">
      <div class="title">New login to your platform</div>
      <div class="subtitle">A user has successfully authenticated. Review the details below.</div>
      <div class="detail-grid">
        <div class="detail-row">
          <span class="detail-label">User Email</span>
          <span class="detail-value">${userEmail}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">IP Address</span>
          <span class="detail-value">${ip}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Timestamp</span>
          <span class="detail-value">${timestamp}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Device / Browser</span>
          <span class="detail-value">${userAgent}</span>
        </div>
        ${location ? `
        <div class="detail-row">
          <span class="detail-label">Location</span>
          <span class="detail-value">${location}</span>
        </div>` : ''}
      </div>
    </div>
    <div class="footer">
      <p>This is an automated security alert from your FintechApp auth system.<br>
      If this looks suspicious, <a href="#">review your security settings</a>.</p>
    </div>
  </div>
</body>
</html>`;

  const mailOptions = {
    from: `"FintechApp Security" <${process.env.ZOHO_USER}>`,
    to: process.env.ADMIN_EMAIL,
    subject: `🔐 Login Alert: ${userEmail} signed in`,
    html,
    text: `Login Alert\n\nUser: ${userEmail}\nIP: ${ip}\nTime: ${timestamp}\nDevice: ${userAgent}`,
  };

  try {
    const info = await transport.sendMail(mailOptions);
    logger.info('Login alert email sent', { messageId: info.messageId, userEmail });
    return true;
  } catch (err) {
    logger.error('Failed to send login alert email', { error: err.message, userEmail });
    // Non-critical: don't throw, login still succeeds
    return false;
  }
};

/**
 * Sends a welcome email on registration.
 */
const sendWelcomeEmail = async ({ userEmail, firstName }) => {
  const transport = getTransporter();

  const mailOptions = {
    from: `"FintechApp" <${process.env.ZOHO_USER}>`,
    to: userEmail,
    subject: 'Welcome to FintechApp — your account is ready',
    html: `
      <div style="font-family: -apple-system, sans-serif; max-width: 500px; margin: 0 auto; padding: 40px 20px;">
        <h2 style="color: #0d1117;">Welcome, ${firstName || 'there'}! 👋</h2>
        <p style="color: #656d76; margin-top: 12px;">Your account has been created successfully. You can now sign in to your dashboard.</p>
        <p style="color: #656d76; margin-top: 24px; font-size: 12px;">If you didn't create this account, please contact support immediately.</p>
      </div>
    `,
  };

  try {
    await transport.sendMail(mailOptions);
    logger.info('Welcome email sent', { userEmail });
  } catch (err) {
    logger.error('Failed to send welcome email', { error: err.message, userEmail });
  }
};

module.exports = { sendLoginAlert, sendWelcomeEmail };