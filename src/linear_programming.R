# CS164 Session 16 — Linear Programming (base R)
# Use: source("src/linear_programming.R"); run_session16()
# Or: Rscript src/linear_programming.R (exports session16_plots.pdf)

vertices3 <- rbind(c(0,6), c(2,2), c(6,0))
vertices5 <- rbind(c(0,0), c(1.5,0), c(1,1), c(0.8,1.2), c(0,1.6))
profit <- function(x,y) 8*x + 6*y

region3 <- function() {
  plot(NA, xlim=c(0,10), ylim=c(0,10), asp=1,
       xlab="x", ylab="y", main="Q3: Feasible region")
  grid(col="gray90")
  polygon(c(0,0,10,10,6,2), c(6,10,10,0,0,2),
          col=adjustcolor("lightgreen", alpha.f=0.45), border=NA)
  segments(0,6,2,2,col="darkgreen",lwd=2.5)
  segments(2,2,6,0,col="darkgreen",lwd=2.5)
  points(vertices3,pch=19)
  text(vertices3[,1],vertices3[,2],c("(0,6)","(2,2)","(6,0)"),
       pos=4,cex=0.8,xpd=TRUE)
}

contours3 <- function(fun, levels, title, minimum=NULL) {
  region3()
  title(main=title)
  grid_x <- seq(0,10,length.out=301)
  z <- outer(grid_x,grid_x,FUN=fun)
  contour(grid_x,grid_x,z,levels=levels,add=TRUE,
          col="steelblue",lwd=1.4,drawlabels=TRUE)
  if (!is.null(minimum)) points(minimum[1],minimum[2],
                               col="red",pch=19,cex=1.5)
}

region4 <- function(L=8) {
  stopifnot(L>=3)
  plot(NA,xlim=c(0,L),ylim=c(0,L),asp=1,xlab="x",ylab="y",
       main=paste("Q4: Feasible region, scale",L))
  grid(col="gray90")
  # y <= (3x-1)/2, y >= x-2, x,y >= 0.
  polygon(c(1/3,2,L,L,(2*L+1)/3),c(0,0,L-2,L,L),
          col=adjustcolor("lightgreen",alpha.f=0.45),border=NA)
  abline(a=-0.5,b=1.5,col="darkgreen",lwd=2)
  abline(a=-2,b=1,col="darkgreen",lwd=2)
  for(v in c(2,6,12,20,32))
    abline(a=v,b=-1,col="steelblue",lty=2)
  segments(1,1,L,L,col="red",lwd=2,lty=3)
  ts <- unique(c(1,floor(L/2),floor(0.75*L)))
  points(ts,ts,col="red",pch=19)
  text(ts,ts,paste0("f=",2*ts),pos=4,cex=0.75)
}

region5 <- function() {
  plot(NA,xlim=c(0,2.2),ylim=c(0,2.2),asp=1,
       xlab="Product x",ylab="Product y",
       main="Q5: Profit maximization")
  grid(col="gray90")
  polygon(vertices5[,1],vertices5[,2],
          col=adjustcolor("lightgreen",alpha.f=0.5),
          border="darkgreen",lwd=2)
  abline(a=2,b=-1,col="orange",lwd=2)     # x+y=2
  abline(a=1.6,b=-0.5,col="purple",lwd=2) # 5x+10y=16
  abline(a=3,b=-2,col="darkgreen",lwd=2) # 2x+y=3
  for(v in c(6,9.6,12,13.6))
    abline(a=v/6,b=-4/3,col="steelblue",lty=2)
  abline(a=14/6,b=-4/3,col="blue",lwd=3,lty=2)
  points(vertices5,pch=19)
  points(1,1,col="red",pch=19,cex=1.6)
  text(1,1,"Maximum = 14",pos=3,col="red",cex=0.8)
  legend("topright",c("x+y=2","5x+10y=16","2x+y=3",
                      "Profit levels","Optimal level"),
         col=c("orange","purple","darkgreen","steelblue","blue"),
         lty=c(1,1,1,2,2),lwd=c(2,2,2,1,3),cex=0.7,bty="n")
}

run_session16 <- function() {
  region3()
  contours3(function(x,y) x+y,c(4,5,6,8,10,12,15),
            "Q3a: x+y; minimum 4 at (2,2)",c(2,2))
  contours3(function(x,y) 3*x+y,c(6,8,12,18,24,30,36),
            "Q3b: 3x+y; minimum 6 at (0,6)",c(0,6))
  contours3(function(x,y) x-y,c(-10,-6,-3,0,3,6,10),
            "Q3c: x-y; no finite extrema")
  contours3(function(x,y) (x-3)^2+(y-3)^2,c(0.5,2,5,10,20,40,60),
            "Q3d: quadratic; minimum 0 at (3,3)",c(3,3))
  contours3(function(x,y) pmax(x,y),c(2,3,4,5,6,8,10),
            "Q3e: max(x,y); minimum 2 at (2,2)",c(2,2))

  region4(8)
  region4(20)
  cat("Q4: every (t,t), t>=1, is feasible and has objective 2t -> infinity.\n")

  region5()
  results <- data.frame(x=vertices5[,1],y=vertices5[,2],
                        profit=profit(vertices5[,1],vertices5[,2]))
  print(results)
  stopifnot(max(results$profit)==14)
  cat("Q5: optimum (1,1); profit 14.\n")
  # KKT certificate for the first three inequalities.
  A <- rbind(c(1,1),c(5,10),c(2,1))
  lambda <- c(4,0,2)
  stopifnot(all(as.vector(t(A)%*%lambda)==c(8,6)))
  cat("KKT multipliers:",lambda,"\n")
}
if (sys.nframe()==0L) {
  pdf("session16_plots.pdf",width=8,height=7,onefile=TRUE)
  run_session16()
  dev.off()
  cat("Saved session16_plots.pdf\n")
}
