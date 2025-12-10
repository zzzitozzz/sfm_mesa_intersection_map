const ContinuousVisualization = function(width, height, context){
    this.draw = function(objects){
        for(const p of objects){
            if(p.Shape == "rect")
                this.drawRectancgle(p.x, p.y, p.w, p.h, p.Color, p.Filled);
            if(p.Shape == "circle")
                this.drawCircle(p.x, p.y, p.r, p.Color, p.Filled);
            if(p.Shape == "ellipse")
                this.drawEllipse(p.x, p.y , p.r, p.Color, p.Filled);
        };
    };

    this.drawCircle = function(x, y, radius, color, fill){
        const cx = x * width;
        const cy = y * height;
        const r = radius;
        context.beginPath();
        context.arc(cx, cy, r, 0, Math.PI * 2, false);
        context.strokeStyle = color;
        context.stroke();
        if(fill){
            context.fillStyle = color;
            context.fill();
        }
    };

    
    this.drawEllipse = function(x, y, radius, color, fill){
        const cx_2 = x * width;
        const cy_2 = y * height;
        // const radiusX = radius * width;
        // const radiusY = radius * height;
        const radiusX = radius * width / 60.;
        const radiusY = radius * height / 60.;
        context.beginPath();
        context.ellipse(cx_2, cy_2, radiusX, radiusY, 0, 0, Math.PI * 2, false);
        context.strokeStyle = color;
        context.stroke();
        if(fill){
            context.fillStyle = color;
            context.fill();
        }

        // context.beginPath();
        // context.ellipse(cx_2, cy_2, radiusX * 0.1, radiusY * 0.1, 0, 0, Math.PI * 2, false);
        // context.strokeStyle = color;
        // context.stroke();

        // if(fill){
        //     context.fillStyle = color;
        //     context.fill();
        // }
        

    };

    // this.drawRectange = function(x, y, w, h, color, fill){
    this.drawRectancgle = function(x, y, w, h, color, fill){
        context.beginPath();
        const dx = w * width;
        const dy = h * height;

        const x0 = (x*width) - 0.5*dx;
        const y0 = (y*height) - 0.5*dy;

        context.strokeStyle = color;
        context.fillStyle = color;
        if(fill)
            context.fillRect(x0, y0, dx, dy);
        else 
            context.rect(x0, y0, dx, dy);
            context.stroke();
    };

    this.resetCanvas = function(){
        context.clearRect(0, 0, width, height);
        context.beginPath()
    };

}

const Simple_Continous_Module = function(canvas_width, canvas_height){

    const canvas = document.createElement("canvas");
    Object.assign(canvas, {
        width: canvas_width,
        height: canvas_height,
        style: 'border: 1px dotted'
    });

    document.getElementById("elements").appendChild(canvas);

        const context = canvas.getContext("2d");
        const canvasDraw = new ContinuousVisualization(canvas_width, canvas_height, context);

        this.render = function(data){
            canvasDraw.resetCanvas();
            canvasDraw.draw(data);
        }

        this.reset = function(){
            canvasDraw.resetCanvas();
        }
}