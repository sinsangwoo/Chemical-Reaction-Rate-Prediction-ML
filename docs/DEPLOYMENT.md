# Cloud Deployment Guide

Complete guide for deploying Chemical Reaction ML Platform to production.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Deploy](#quick-deploy)
3. [Platform Options](#platform-options)
4. [Detailed Guides](#detailed-guides)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer / CDN             │
│         (Cloudflare / AWS)              │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐  ┌───────▼──────┐
│   Frontend   │  │   Backend    │
│ (Vercel/S3)  │  │  (Railway)   │
└──────────────┘  └───────┬──────┘
                          │
                  ┌───────▼──────┐
                  │  PostgreSQL  │
                  │  (Railway)   │
                  └──────────────┘
```

### Cost Estimation

| Tier | Monthly Cost | Users | Requests/mo |
|------|-------------|-------|-------------|
| **Free** | $0 | <100 | <10K |
| **Hobby** | $20-30 | <1K | <100K |
| **Production** | $100-200 | <10K | <1M |
| **Enterprise** | $500+ | Unlimited | Unlimited |

---

## Quick Deploy

### Option 1: Railway (Easiest)

**1-Click Deploy**:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

**Manual Deploy**:

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Create project
railway init

# 4. Add PostgreSQL
railway add
# Select: PostgreSQL

# 5. Deploy backend
cd api
railway up

# 6. Deploy frontend
cd ../frontend
railway up

# Done! Get URL:
railway domain
```

### Option 2: Vercel + Render

**Frontend (Vercel)**:

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Deploy
cd frontend
vercel --prod
```

**Backend (Render)**:

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add PostgreSQL database
6. Deploy!

### Option 3: AWS (Full Control)

See [AWS Deployment](#aws-deployment) section below.

---

## Platform Options

### Recommended Stack

**For Beginners**:
- Frontend: Vercel (free)
- Backend: Railway ($5/mo)
- Database: Railway PostgreSQL (included)
- **Total**: ~$5/month

**For Production**:
- Frontend: Vercel Pro ($20/mo) or AWS CloudFront + S3
- Backend: Railway Pro ($20/mo) or AWS ECS
- Database: Railway PostgreSQL or AWS RDS
- **Total**: ~$40-100/month

**For Enterprise**:
- Frontend: AWS CloudFront + S3
- Backend: AWS ECS Fargate with auto-scaling
- Database: AWS RDS Multi-AZ
- **Total**: $200-500/month

### Platform Comparison

| Feature | Railway | Render | Vercel | AWS |
|---------|---------|--------|--------|-----|
| **Ease** | ★★★★★ | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| **Cost** | $5-20 | $7-25 | $0-20 | $20-500 |
| **Speed** | Fast | Fast | Fastest | Fast |
| **Scale** | Good | Good | Excellent | Unlimited |
| **Support** | Good | Good | Excellent | Enterprise |

---

## Detailed Guides

### Railway Deployment

#### Prerequisites

```bash
npm i -g @railway/cli
railway login
```

#### Step-by-Step

**1. Create Project**

```bash
railway init
# Name: chemical-ml-platform
```

**2. Add Database**

```bash
railway add
# Select: PostgreSQL

# Get DATABASE_URL
railway variables
# Copy DATABASE_URL value
```

**3. Set Environment Variables**

```bash
railway variables set SECRET_KEY=$(openssl rand -hex 32)
railway variables set DATABASE_URL=<postgres-url>
```

**4. Deploy Backend**

```bash
cd api
railway up

# Get backend URL
railway domain
# Example: chemical-ml-api.up.railway.app
```

**5. Deploy Frontend**

```bash
cd ../frontend

# Set API URL
echo "VITE_API_URL=https://chemical-ml-api.up.railway.app" > .env.production

railway up

# Get frontend URL
railway domain
```

**6. Configure Custom Domain (Optional)**

```bash
railway domain add yourdomain.com
```

#### Railway Configuration Files

**railway.json** (Backend):

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**railway.json** (Frontend):

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "npx serve -s dist -l $PORT"
  }
}
```

---

### Vercel Deployment

#### Frontend

**1. Install Vercel CLI**

```bash
npm i -g vercel
```

**2. Configure**

**vercel.json**:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://your-backend-url.com/:path*"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

**3. Deploy**

```bash
cd frontend
vercel --prod
```

**4. Environment Variables**

Add in Vercel dashboard:
- `VITE_API_URL`: Your backend URL

---

### AWS Deployment

#### Architecture

```
CloudFront (CDN)
    ↓
S3 (Frontend) + API Gateway
    ↓
ECS Fargate (Backend)
    ↓
RDS PostgreSQL (Database)
```

#### Step-by-Step

**1. Setup AWS CLI**

```bash
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1)
```

**2. Create S3 Bucket (Frontend)**

```bash
# Create bucket
aws s3 mb s3://chemical-ml-frontend

# Build frontend
cd frontend
npm run build

# Upload
aws s3 sync dist/ s3://chemical-ml-frontend --delete

# Enable website hosting
aws s3 website s3://chemical-ml-frontend \
  --index-document index.html \
  --error-document index.html
```

**3. Setup CloudFront (CDN)**

```bash
aws cloudfront create-distribution \
  --origin-domain-name chemical-ml-frontend.s3.amazonaws.com \
  --default-root-object index.html

# Get distribution URL
aws cloudfront list-distributions
```

**4. Create RDS Database**

```bash
aws rds create-db-instance \
  --db-instance-identifier chemical-ml-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password <your-password> \
  --allocated-storage 20

# Get endpoint
aws rds describe-db-instances \
  --db-instance-identifier chemical-ml-db \
  --query 'DBInstances[0].Endpoint.Address'
```

**5. Deploy Backend to ECS**

```bash
# Build and push Docker image
cd api
aws ecr create-repository --repository-name chemical-ml-api

# Get login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t chemical-ml-api .
docker tag chemical-ml-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/chemical-ml-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/chemical-ml-api:latest

# Create ECS cluster
aws ecs create-cluster --cluster-name chemical-ml-cluster

# Create task definition and service (see AWS docs)
```

#### AWS Cost Optimization

```bash
# Use t3.micro for dev ($10/mo)
# Use t3.small for production ($20/mo)
# Enable auto-scaling to reduce costs

# Estimated monthly costs:
# - S3 + CloudFront: $5-20
# - ECS Fargate: $20-50
# - RDS t3.micro: $15-25
# Total: $40-95/month
```

---

## Configuration

### Environment Variables

**Backend (.env)**:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Security
SECRET_KEY=your-secret-key-here

# CORS (production frontend URL)
CORS_ORIGINS=https://yourdomain.com

# Optional
ENVIRONMENT=production
LOG_LEVEL=info
```

**Frontend (.env.production)**:

```bash
VITE_API_URL=https://api.yourdomain.com
```

### SSL/TLS Certificates

**Option 1: Let's Encrypt (Free)**

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

**Option 2: Cloudflare (Free)**

1. Add your domain to Cloudflare
2. Update nameservers
3. Enable SSL (Full or Full Strict)
4. Done! Auto-managed certificates

**Option 3: AWS Certificate Manager (Free)**

```bash
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS
```

---

## Monitoring

### Health Checks

**Setup monitoring endpoints**:

```python
# api/main.py
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": check_db_connection(),
        "timestamp": datetime.now().isoformat()
    }
```

**Monitor with cron**:

```bash
# Check every 5 minutes
*/5 * * * * curl -f https://api.yourdomain.com/health || echo "API down!"
```

### Logging

**Setup structured logging**:

```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format=json.dumps({
        "time": "%(asctime)s",
        "level": "%(levelname)s",
        "message": "%(message)s"
    })
)
```

**View logs**:

```bash
# Railway
railway logs

# AWS CloudWatch
aws logs tail /ecs/chemical-ml-api --follow

# Render
# View in dashboard
```

### Uptime Monitoring

**Services**:
- **UptimeRobot** (free): https://uptimerobot.com
- **Better Uptime** (free tier): https://betteruptime.com
- **AWS CloudWatch** (paid): Built-in

**Setup example** (UptimeRobot):

1. Sign up free
2. Add monitor:
   - Type: HTTPS
   - URL: https://api.yourdomain.com/health
   - Interval: 5 minutes
3. Get alerts via email/Slack/Discord

---

## Troubleshooting

### Common Issues

**1. CORS Errors**

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. Database Connection Issues**

```bash
# Check DATABASE_URL format
postgresql://user:password@host:5432/dbname

# Test connection
psql $DATABASE_URL
```

**3. Build Failures**

```bash
# Clear cache
rm -rf node_modules dist
npm install
npm run build

# Check logs
railway logs
```

**4. 502 Bad Gateway**

- Check backend is running
- Verify port configuration
- Check health endpoint
- Review logs

**5. Slow Performance**

- Enable CDN (Cloudflare)
- Optimize database queries
- Add caching (Redis)
- Scale up instances

---

## Security Checklist

- [ ] HTTPS enabled (SSL certificate)
- [ ] Environment variables (no hardcoded secrets)
- [ ] CORS configured (specific origins)
- [ ] Rate limiting enabled
- [ ] Database backups automated
- [ ] Logging enabled
- [ ] Monitoring alerts setup
- [ ] Security headers configured
- [ ] Dependencies updated
- [ ] Authentication required for sensitive endpoints

---

## Scaling Guide

### Horizontal Scaling

**Railway**:

```bash
# Scale to 2 instances
railway scale --replicas 2
```

**AWS ECS**:

```bash
# Update service
aws ecs update-service \
  --cluster chemical-ml-cluster \
  --service chemical-ml-api \
  --desired-count 3
```

### Vertical Scaling

**Railway**:

- Upgrade to Pro plan
- More CPU/RAM automatically allocated

**AWS**:

```bash
# Change instance type
aws ecs update-service \
  --cluster chemical-ml-cluster \
  --service chemical-ml-api \
  --task-definition chemical-ml-api:2  # Updated task with larger instance
```

### Database Scaling

**Read Replicas**:

```bash
aws rds create-db-instance-read-replica \
  --db-instance-identifier chemical-ml-db-replica \
  --source-db-instance-identifier chemical-ml-db
```

**Connection Pooling**:

```python
# Use pgbouncer or SQLAlchemy pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0
)
```

---

## Backup & Recovery

### Automated Backups

**Railway**:

- Automatic daily backups (Pro plan)
- 7-day retention

**AWS RDS**:

```bash
# Enable automated backups
aws rds modify-db-instance \
  --db-instance-identifier chemical-ml-db \
  --backup-retention-period 7
```

### Manual Backup

```bash
# PostgreSQL dump
pg_dump $DATABASE_URL > backup.sql

# Restore
psql $DATABASE_URL < backup.sql
```

---

## Support

**Issues?**

1. Check [Troubleshooting](#troubleshooting) section
2. Review platform docs:
   - Railway: https://docs.railway.app
   - Vercel: https://vercel.com/docs
   - AWS: https://docs.aws.amazon.com
3. Open GitHub issue

**Need help?**

- GitHub Issues: Report bugs
- Discussions: Ask questions
- Discord/Slack: Community support (if available)
