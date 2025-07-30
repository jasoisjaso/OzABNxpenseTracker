FROM nginx:alpine

# Copy the custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY tax_guide.html /usr/share/nginx/html

# Copy the frontend application files
COPY index.html /usr/share/nginx/html/index.html

# Ensure the nginx user owns the web root, which can prevent permission issues.
RUN chown -R nginx:nginx /usr/share/nginx/html