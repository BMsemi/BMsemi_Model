Network Yolo26nano {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2}
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640}
        Dataflow {
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(Sz(R),Sz(R)) R;
            TemporalMap(Sz(S),Sz(S)) S;
            TemporalMap(Sz(R),1) Y;
            TemporalMap(Sz(S),1) X;
        }
    }
}
    % Layer Conv2D
    %     Type: Conv
    %     Stride { X: 2, Y: 2}
    %     Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 320, X: 320}
    %     Dataflow {
    %         SpatialMap(8,8) K;
    %         TemporalMap(4,4) C;
    %         TemporalMap(Sz(R),Sz(R)) R;
    %         TemporalMap(Sz(S),Sz(S)) S;
    %         TemporalMap(Sz(R),1)) Y;
    %         TemporalMap(Sz(S),1)) X;
    %         cluster(4,P);
    %         SpatialMap(4,4) C;
    %         TemporalMap(Sz(R),Sz(R)) R;
    %         TemporalMap(Sz(S),Sz(S)) S;
    %         TemporalMap(Sz(R),1)) Y;
    %         TemporalMap(Sz(S),1)) X;
    %     }
    % }

    % Layer Conv2D
    %     Type: Conv
    %     Stride { X: 1, Y: 1}
    %     Dimensions { K: 32, C: 32, R: 1, S: 1, Y: 160, X: 160}
    %     Dataflow {
    %         SpatialMap(8,8) K;
    %         TemporalMap(4,4) C;
    %         TemporalMap(Sz(R),Sz(R)) R;
    %         TemporalMap(Sz(S),Sz(S)) S;
    %         TemporalMap(Sz(R),1)) Y;
    %         TemporalMap(Sz(S),1)) X;
    %         Cluster(8,P);
    %         SpatialMap(4,4) C;
    %         TemporalMap(Sz(R),Sz(R)) Y;
    %         TemporalMap(Sz(S),Sz(S)) X;
    %         TemporalMap(Sz(R),1)) R;
    %         TemporalMap(Sz(S),1)) S;
    %     }
    % }

    % Layer Conv2D 
    %     Type: Conv
    %     Stride { X: 1, Y: 1}
    %     Dimensions { K: 8, C: 16, R: 3, S: 3, Y: 160, X: 160}
    %     Dataflow {
    %         SpatialMap(2,2) K;
    %         TemporalMap(4,4) C;
    %         TemporalMap(Sz(R),Sz(R)) R;
    %         TemporalMap(Sz(S),Sz(S)) S;
    %         TemporalMap(Sz(R),1)) Y;
    %         TemporalMap(Sz(S),1)) X;
    %     }
    % }

    % Layer Conv2D
    %     Type: Conv
    %     Stride { X: 1, Y: 1}
    %     Dimensions { K: 16, C: 8, R: 3, S: 3, Y: 160, X: 160}
    %     Dataflow {
    %         SpatialMap()



            
    %     }




    
    


    






    

